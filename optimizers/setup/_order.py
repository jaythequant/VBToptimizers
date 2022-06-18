import vectorbt as vbt
import pandas as pd
import gc

from .lqe_setup import * 
from .statistics import extract_duration, extract_wr, number_of_trades


def simulate_from_order_func(
    close_data, open_data, period, upper, lower, exits, burnin=500, delta=1e-5, vt=1, 
    mode="Kalman", cash=100_000, commission=0.0008, slippage=0.0010, order_size=0.10, 
    freq="d"
):
    """Simulate pairs trading strategy with multiple signal strategy optionality"""
    return vbt.Portfolio.from_order_func(
        close_data,
        order_func_nb, 
        open_data.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=pre_group_func_nb, 
        pre_group_args=(
            period, 
            upper, 
            lower, 
            exits, 
            delta, 
            vt, 
            order_size, 
            burnin,
        ),
        pre_segment_func_nb=pre_segment_func_nb, 
        pre_segment_args=(mode,),
        fill_pos_record=False,  # a bit faster
        init_cash=cash,
        cash_sharing=True, 
        group_by=True,
        freq=freq
    )


def simulate_mult_from_order_func(
    close_prices, open_prices, params, commission=0.0008, slippage=0.0005, mode="Kalman", 
    cash=100_000, order_size=0.10, burnin=500, freq="m", interval="minutes",
):
    """Backtest multiple parameter combinations using VectorBT's `vbt.Portfolio.from_order_func`"""
    # Generate multiIndex columns
    param_tuples = list(zip(*params.values()))
    param_columns = pd.MultiIndex.from_tuples(param_tuples, names=params.keys())

    # We need two price columns per param combination
    vbt_close_price_mult = close_prices.vbt.tile(len(param_columns), keys=param_columns)
    vbt_open_price_mult = open_prices.vbt.tile(len(param_columns), keys=param_columns)

    # Simulate the portfolio and return portfolio object
    pf = vbt.Portfolio.from_order_func(
        vbt_close_price_mult,
        order_func_nb, 
        vbt_open_price_mult.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=pre_group_func_nb,
        pre_group_args=(
            np.array(params["period"]),
            np.array(params["upper"]), 
            np.array(params["lower"]), 
            np.array(params["exit"]), 
            np.array(params["delta"]), 
            np.array(params["vt"]), 
            order_size,
            burnin,
        ),
        pre_segment_func_nb=pre_segment_func_nb,
        pre_segment_args=(mode,),
        fill_pos_record=False,
        init_cash=cash,
        cash_sharing=True, 
        group_by=param_columns.names,
        freq=freq,
    )

    # Append results of each param comb to CSV file
    total_return = pf.total_return()    # Extract total return for each param
    wr = extract_wr(pf) # Extract win rate on net long-short trade for each param
    dur = extract_duration(pf, interval) # Extract median trade duration in hours
    trades = number_of_trades(pf)
    # Append params results to CSV for later analysis
    res = pd.concat([total_return, wr, dur, trades], axis=1)
    gc.collect()
    return res
