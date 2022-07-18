import vectorbt as vbt
import pandas as pd
import gc

from .lqe_setup import * 
from .statistics import extract_duration, extract_wr, number_of_trades


def simulate_from_order_func(
    close_data:pd.DataFrame, open_data:pd.DataFrame, period:float, upper:float,
    lower:float, exit:float, burnin:int=500, delta:float=1e-5, vt:float=1.0, 
    mode:str="default", cash:int=100_000, commission:float=0.0008, 
    slippage:float=0.0010, order_size:float=0.10, freq:None or str=None, hedge:str="dollar",
):
    """Highly configurable pair trade backtest environment built on vectorBT

    Parameters
    ----------
    close_data : pd.DataFrame
        DataFrame containing close data with datetime index of uniform frequency.
        The DataFrame must contain two columns representing two assets. The first
        column will be treated as the X asset while the second will be treated as
        the y asset.
    open_data : pd.DataFrame
        DataFrame containing open data with datetime index of uniform frequency.
        The DataFrame must contain two columns representing two assets. The first
        column will be treated as the X asset while the second will be treated as
        the y asset.
    period : float
        Number of bars data to use as a lookback period for the strategy's rolling
        statistics.
    upper : float
        Upper threshold above which to trigger a short-the-spread trade. This upper
        threshold will be compared to a rolling z-score calculated across `period`
        bars of data and should be thought of as a number of standard deviations
        divergence required to make a trade.
    lower : float
        Lower threshold below which to trigger a long-the-spread trade. This lower
        threshold will be compared to a rolling z-score calculated across `period`
        bars of data and should be thought of as a number of standard deviations
        divergence required to make a trade.
    exit : float
        If a long-short position in opened (i.e. we are long or short the spread),
        then we use the `exit` parameter to determine when to unwind our positions.
        When `np.abs(z_t) < exit` we close the position.
    burnin : int, optional
        Kalman Filter's (i.e. Linear Quadratic Estimators or LQEs) require a certain
        number of observations to calibrate to an appropriately balanced theta term. 
        This variable dictates how many iterations (i.e. how many bars) are required 
        to pass before any trading signals are recognized. Optimal burnin length 
        various significantly across data sets and requires outside testing to optimize.
    delta : float, optional
        Delta is one of two parameters required to calibrate the Kalman filter. Delta
        must be found through outside optimization although a default value is given 
        here. Be aware that delta has an extremely high degree of importance to strategy 
        success. 
    vt : float, optional
        The second of the two parameters required for the Kalman filter. Like `delta`, 
        `vt` must be found through strategy optimization despite a default value being 
        given. `vt` is even more important than `delta` to optimize.
    mode : str, optional
        This strategy may be run using several price transformations or no price 
        transformation. By default, the strategy is run on raw asset prices. If 
        `mode=cummlog`, Kalman filter estimates will be made by regressing cummulative 
        logarithmic returns on each other.
    cash : int, optional
        Starting cash for the portfolio object in vectorBT. See docs for `vbt.Portfolio` 
        for details
    commission : float, optional
        Per trade commission as a percentage. This figure will be applied to all trades 
        executed during the backtest. See `vbt.Portfolio` for details.
    slippage : float, optional
        Per trade slippage as a percentage. Execution prices will randomly increase 
        or decrease during execution to simulate real world price variability. See 
        `vbt.Portfolio` for details. 
    order_size : float, optional
        Percentage of portfolio value to allocate per side of the trade. Currently 
        `order_size` is a fixed amount based on the starting portfolio value as 
        dictated by `cash`.
    freq : str, optional
        Frequency at which price data should be recorded. `freq` should match the 
        interval of `close_data` and `open_data` indices. If `freq=None` [default]
        then the function will attempt to parse interval from `close_data.index` 
        intervals.
    hedge : str, optional
        Specify how the strategy should size trades. If `hedge="dollar` [default],
        trades will be un-hedged and simply execute 1:1 long-short. If `hedge="beta"`,
        trades will be beta hedged using theta[0] (equivalent to beta_1 in 
        regression).

    Returns
    -------
        Portfolio object with a large amount of review data regarding the simulation
        results. See https://vectorbt.dev/ for information.
    """
    return vbt.Portfolio.from_order_func(
        close_data,
        order_func_nb, 
        open_data.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=pre_group_func_nb, 
        pre_group_args=(
            period, 
            upper, 
            lower, 
            exit, 
            delta, 
            vt, 
            order_size, 
            burnin,
        ),
        pre_segment_func_nb=pre_segment_func_nb, 
        pre_segment_args=(mode, hedge,),
        fill_pos_record=False,  # a bit faster
        init_cash=cash,
        cash_sharing=True, 
        group_by=True,
        freq=freq
    )


def simulate_mult_from_order_func(
    close_prices, open_prices, params, commission=0.0008, slippage=0.0005, mode="default", 
    cash=100_000, order_size=0.10, burnin=500, freq="m", interval="minutes", hedge="dollar",
):
    """Backtest multiple parameter combinations as cartesian product"""
    # Generate multiIndex columns
    param_product = vbt.utils.params.create_param_product(list(params.values())) # Somehow this got lost at one point

    for idx, prod in enumerate(param_product):  # vbt create_param_product creates rounding errors
        prod = [round(num, 15) for num in prod]  # In response, we clean up the numbers
        param_product[idx] = prod               # Then reconstruct the param_product var

    param_tuples = list(zip(*param_product))
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
        pre_segment_args=(mode, hedge,),
        fill_pos_record=False,
        init_cash=cash,
        cash_sharing=True, 
        group_by=param_columns.names,
        freq=freq,
    )

    # Append results of each param comb to CSV file
    # total_return = pf.total_return()    # Extract total return for each param
    # wr = extract_wr(pf) # Extract win rate on net long-short trade for each param
    # dur = extract_duration(pf, interval) # Extract median trade duration in hours
    # trades = number_of_trades(pf)
    # Append params results to CSV for later analysis
    # res = pd.concat([total_return, wr, dur, trades], axis=1)
    gc.collect()
    return pf


def simulate_batch_from_order_func(
    close_prices, open_prices, params, commission=0.0008, slippage=0.0005, mode="default", 
    cash=100_000, order_size=0.10, burnin=500, freq="m", interval="minutes", hedge="dollar",
):
    """Backtest pre-batched param sets [Param sets must be pre-defined]"""
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
        pre_segment_args=(mode, hedge,),
        fill_pos_record=False,
        init_cash=cash,
        cash_sharing=True, 
        group_by=param_columns.names,
        freq=freq,
    )

    # Append results of each param comb to CSV file
    # total_return = pf.total_return()    # Extract total return for each param
    # wr = extract_wr(pf) # Extract win rate on net long-short trade for each param
    # dur = extract_duration(pf, interval) # Extract median trade duration in hours
    # trades = number_of_trades(pf)
    # Append params results to CSV for later analysis
    # res = pd.concat([total_return, wr, dur, trades], axis=1)
    gc.collect()
    return pf
