import pandas as pd
import timedelta as td
import numpy as np


def extract_wr(pf) -> float:
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
    
    dur_ser = pd.Series(dur_dict.values(), index=dur_dict.keys(), name="Duration")

    return dur_ser


def generate_random_sample(n_iter=100):
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
