import pandas as pd
import timedelta as td
import numpy as np


def extract_wr(pf) -> pd.Series:
    """
    Extracts win rate from vbt portfolio object as % of net long-short trade

    :param pf: VectorBT portfolio object
    :return wr_ser: pandas Series with index as param combination and
        and win rate percentage of net long-short pair trade as values
    """
    # Extract trade records from vbt portfolio object
    trades = pf.trades.records_readable
    # Trade records column is a tuple with asset included. 
    # Because we want our net long-short WR not our per asset WR,
    # we have to split the asset out of the column.
    trades["Column"] = trades["Column"].str[:-1]
    # The new trades.Column is just parameter groups without 
    # asset dividing the parameter group which means that the
    # groupby function appropriate divides our trade records into
    # hyperparameter groups.
    g = trades.groupby("Column")

    wr_dict = {}

    for idx, gr in g:
        # Sum PnL for each timestamp to get net long-short PnL
        net = gr.groupby("Entry Timestamp").agg({"PnL": sum})
        win = net.PnL > 0 # Bool mask to select winning trades
        wr = net[win].shape[0] / net.shape[0] # Winning trades / total trades
        wr_dict[idx] = wr # param group = key; % wr = value

    # Wrap our dictionary in pandas Series for ease of analysis
    wr_ser = pd.Series(wr_dict.values(), index=wr_dict.keys(), name="Win Rate")
    
    return wr_ser


def extract_duration(pf, interval) -> int:
    """
    Extracts trade duration from vbt portfolio object as number of hours.
    For detailed notes on each step of this function see notation 
        for extract_wr.

    :param pf: VectorBT portfolio object
    :return dur_ser: pandas Series with index as param combination and
        and median hours in any given trade
    """
    trades = pf.trades.records_readable
    trades["Column"] = trades["Column"].str[:-1]
    g = trades.groupby("Column")

    dur_dict = {}

    for idx, gr in g:
        # Note drop duplicates is needed as each trade registers twice
        # because there is a long and short trade placed simultaneously.
        entries = pd.to_datetime(gr["Entry Timestamp"]).drop_duplicates()
        exits = pd.to_datetime(gr["Exit Timestamp"]).drop_duplicates()
        delta = exits - entries
        dur = getattr(td.Timedelta(delta.median()).total, interval)
        dur_dict[idx] = dur
    
    dur_ser = pd.Series(dur_dict.values(), index=dur_dict.keys(), name="duration")

    return dur_ser


def calculate_profit_ratio(pf, median=True, handle_inf=10) -> pd.Series:
    """Calculates the profit ratio of median profit to median loss per trade"""
    trades = pf.trades.records_readable
    trades["Column"] = trades["Column"].str[:-1]
    g = trades.groupby("Column")

    profit_ratio_dict = {}

    for idx, gr in g:
        net = gr.groupby("Entry Timestamp").agg({"PnL": sum})
        if median:
            profit = net[net > 0].median()
            loss = np.abs(net[net < 0].median())
        else:
            profit = net[net > 0].mean()
            loss = np.abs(net[net < 0].mean())
        ratio = profit / loss

        if loss.isna().any():
            ratio = np.inf
        if profit.isna().any():
            ratio = -np.inf        
        profit_ratio_dict[idx] = float(ratio)

        ser = pd.Series(
            profit_ratio_dict.values(), 
            index=profit_ratio_dict.keys(), 
            name="profit_ratio"
        ).replace([np.inf, -np.inf], handle_inf)

    return ser


def generate_random_sample(n_iter=100, require_unique=True):
    """Generate a set of random samples"""
    samples = [] 

    for _ in range(n_iter):
        random_sample = {
            "period": round(np.random.uniform(low=30, high=60), 0) * 1_000,
            "upper": round(np.random.uniform(low=2.0, high=5.1), 1),
            "lower": round(np.random.uniform(low=2.0, high=5.1), 1) * -1,
            "exit": round(np.random.uniform(low=0.3, high=2.0), 1),
            "delta": round(np.random.choice(0.1 ** np.arange(1, 10, dtype=np.float_)), 10),
            "vt": round(np.random.uniform(low=0.1, high=1.0), 1),
        }
        samples.append(random_sample)

    return samples


def score_results(test_results):
    """Calculate the average win rate for cross validated test results"""
    wrs = [] # Empty list for storing trade win rates

    for pf in test_results:
        wr = pf.trades.records_readable.groupby("Entry Timestamp")["PnL"].sum()
        try:
            wrs.append(wr[wr > 0].shape[0] / wr.shape[0])
        except ZeroDivisionError:
            wrs.append(0)

    return np.mean(wrs)


def return_results(test_results):
    """Extract average total return from cross validated test results"""
    rets = [] # List for storing total return information

    for pf in test_results:
        df = pf.orders.records_readable
        df["val"] = df["Size"] * df["Price"]
        avg_v = df.groupby("Timestamp")["val"].sum().mean()
        med_ret = pf.trades.records_readable.groupby("Entry Timestamp")["PnL"].sum().describe()["50%"]
        r = med_ret / avg_v
        rets.append(r)
    
    return np.mean(rets)


def number_of_trades(pf) -> pd.Series:
    """Extracts number of trades from multi-combination portfolio object"""
    trades = pf.trades.records_readable
    trades["Column"] = trades["Column"].str[:-1]
    g = trades.groupby("Column")

    num_trades = {}

    for idx, gr in g:
        grouped_trades = gr.groupby("Entry Timestamp").sum()
        trade_count = grouped_trades.shape[0] / 2
        num_trades[idx] = trade_count

    trades_ser = pd.Series(num_trades.values(), index=num_trades.keys(), name="trade_count")

    return trades_ser


def _weighted_average(df:pd.DataFrame) -> pd.Series:
    """Calculate the weighted-average win rate across folds"""
    g = df.groupby(by=df.columns, axis=1)       # Group by column
    trades = g.get_group("trade_count")         # Extract trade_count
    wr = g.get_group("Win Rate")                # Extract win rate
    group_len = trades.shape[1]                 # Grab the group lengths
    wr.columns = list(range(0, group_len))      # Rename to unique values
    trades.columns = list(range(0, group_len))    
    weighted = (trades * wr).sum(axis=1)        # Sum weighted win rate
    total_trades = trades.sum(axis=1)           # Sum total weights
    weighted_results = pd.Series(
        weighted / total_trades, 
        name="Weighted Average",
    )
    return weighted_results


def _calculate_mse(train_df, validate_df):
    """Return MSE and STD of errors in training versus validation win rates"""
    train_groups = train_df.groupby(by=train_df.columns, axis=1)
    validate_groups = validate_df.groupby(by=validate_df.columns, axis=1)
    train_wr = train_groups.get_group("Win Rate")
    validate_wr = validate_groups.get_group("Win Rate")
    train_wr.columns = list(range(0, train_wr.shape[1]))
    validate_wr.columns = list(range(0, validate_wr.shape[1]))
    sq_error = (validate_wr - train_wr) ** 2
    mse = pd.Series(sq_error.mean(axis=1), name="MSE")
    std = pd.Series(sq_error.T.describe().T["std"], name="STD")
    return mse, std
