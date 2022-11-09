import gc
import vectorbt as vbt
import pandas as pd
import numpy as np
from .strategies.lqe import lqe_pre_group_func_nb, lqe_pre_segment_func_nb
from .strategies.lqev2 import pre_group_func_nb, pre_segment_func_nb
from .strategies.rollingols import ols_pre_group_func_nb, ols_pre_segment_func_nb
from .strategies.components.preprocessors import cummulative_return_transform, log_return_transform
from .strategies.components.statistics import (
    calculate_profit_ratio, extract_duration, extract_wr, 
    number_of_trades, custom_sharpe_ratio, custom_sortino_ratio,
)
from .strategies.components.orders import order_func_nb


############################################################################
##                Linear Quadratic Estimator Models                       ##
############################################################################


def simulate_lqe_model(
    close_data:pd.DataFrame, open_data:pd.DataFrame, period:float, upper:float,
    lower:float, exit:float, burnin:int=500, delta:float=1e-5, vt:float=1.0, 
    transformation:str=None, cash:int=100_000, commission:float=0.0008, 
    slippage:float=0.0010, order_size:float=0.10, freq:str=None, 
    hedge:str="beta", standard_score='zscore', seed=np.array([]),
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
    transformation : str, optional
        This strategy may be run using several price transformations or no price 
        transformation. By default, the strategy is run on raw asset prices. If 
        `transformation=cummlog`, Kalman filter estimates will be made by regressing cummulative 
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
    seed : np.array, optional
        Set a seed using historic data.

    Returns
    -------
    vectorbt.portfolio.base.Portfolio
        Portfolio object with a large amount of review data regarding the simulation
        results. See https://vectorbt.dev/ for information.
    """
    lower = -lower if lower > 0 else lower
    seed = transform_seed(seed, close_data.shape, transformation)
    return vbt.Portfolio.from_order_func(
        close_data,
        order_func_nb, 
        open_data.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=lqe_pre_group_func_nb, 
        pre_group_args=(
            period, 
            upper, 
            lower, 
            exit, 
            delta, 
            vt, 
            order_size, 
            burnin,
            seed,
        ),
        pre_segment_func_nb=lqe_pre_segment_func_nb, 
        pre_segment_args=(transformation, hedge, standard_score),
        fill_pos_record=False,  # a bit faster
        init_cash=cash,
        cash_sharing=True, 
        group_by=True,
        freq=freq
    )


def simulate_mult_lqe_model(
    close_data:pd.DataFrame, open_data:pd.DataFrame, params:dict,
    commission:float=0.0008, slippage:float=0.0005, transformation:str=None,
    cash:int=100_000, order_size:float=0.10, burnin:int=500, freq:str="h",
    hedge:str="beta", standard_score='zscore', seed=np.array([])
):
    """Simultaneously backtest large set of parameter combinations"""
    seed = transform_seed(seed, close_data.shape, transformation)
    # Generate multiIndex columns
    param_product = vbt.utils.params.create_param_product(list(params.values()))

    for idx, prod in enumerate(param_product):  # vbt create_param_product creates rounding errors
        prod = [round(num, 11) for num in prod]  # In response, we clean up the numbers
        param_product[idx] = prod               # Then reconstruct the param_product var

    param_tuples = list(zip(*param_product))
    param_columns = pd.MultiIndex.from_tuples(param_tuples, names=params.keys())

    # We need two price columns per param combination
    vbt_close_price_mult = close_data.vbt.tile(len(param_columns), keys=param_columns)
    vbt_open_price_mult = open_data.vbt.tile(len(param_columns), keys=param_columns)

    # Simulate the portfolio and return portfolio object
    pf = vbt.Portfolio.from_order_func(
        vbt_close_price_mult,
        order_func_nb, 
        vbt_open_price_mult.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=lqe_pre_group_func_nb,
        pre_group_args=(
            np.array(param_product[0]),
            np.array(param_product[1]),
            np.array(param_product[2]),
            np.array(param_product[3]),
            np.array(param_product[4]),
            np.array(param_product[5]),
            order_size,
            burnin,
        ),
        pre_segment_func_nb=lqe_pre_segment_func_nb,
        pre_segment_args=(transformation, hedge, standard_score),
        fill_pos_record=False,
        init_cash=cash,
        cash_sharing=True, 
        group_by=param_columns.names,
        freq=freq,
    )
    return pf


def simulate_batch_lqe_model(
    close_data:pd.DataFrame, open_data:pd.DataFrame, params:dict, 
    commission:float=0.0008, slippage:float=0.0005, transformation:str=None, 
    cash:int=100_000, order_size:float=0.10, burnin:int=500, 
    freq:str=None, interval:str="minutes", hedge:str="beta",
    rf=0.00, standard_score='zscore', seed=np.array([]),
):
    """Backtest batched param sets [Param sets must be pre-defined]"""
    seed = transform_seed(seed, close_data.shape, transformation)
    # Generate multiIndex columns
    param_tuples = list(zip(*params.values()))
    param_columns = pd.MultiIndex.from_tuples(param_tuples, names=params.keys())

    # We need two price columns per param combination
    vbt_close_price_mult = close_data.vbt.tile(len(param_columns), keys=param_columns)
    vbt_open_price_mult = open_data.vbt.tile(len(param_columns), keys=param_columns)

    # Simulate the portfolio and return portfolio object
    pf = vbt.Portfolio.from_order_func(
        vbt_close_price_mult,
        order_func_nb, 
        vbt_open_price_mult.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=lqe_pre_group_func_nb,
        pre_group_args=(
            np.array(params["period"]),
            np.array(params["upper"]), 
            np.array(params["lower"]), 
            np.array(params["exit"]), 
            np.array(params["delta"]), 
            np.array(params["vt"]), 
            order_size,
            burnin,
            seed,
        ),
        pre_segment_func_nb=lqe_pre_segment_func_nb,
        pre_segment_args=(transformation, hedge, standard_score),
        fill_pos_record=False,
        init_cash=cash,
        cash_sharing=True, 
        group_by=param_columns.names,
        freq=freq,
    )
    res = _analyze_results(pf, interval, burnin, rf)
    gc.collect()
    return res

def simulate_batch_from_order_func_low_param(
    close_data:pd.DataFrame, open_data:pd.DataFrame, params:dict, 
    commission:float=0.0008, slippage:float=0.0005, transformation:str="default", 
    cash:int=100_000, order_size:float=0.10, burnin:int=500, 
    freq:None or str=None, interval:str="minutes", hedge:str="dollar",
    rf=0.05,
):
    """Backtest batched param sets [Param sets must be pre-defined]"""
    # Generate multiIndex columns
    param_tuples = list(zip(*params.values()))
    param_columns = pd.MultiIndex.from_tuples(param_tuples, names=params.keys())

    # We need two price columns per param combination
    vbt_close_price_mult = close_data.vbt.tile(len(param_columns), keys=param_columns)
    vbt_open_price_mult = open_data.vbt.tile(len(param_columns), keys=param_columns)

    # Simulate the portfolio and return portfolio object
    pf = vbt.Portfolio.from_order_func(
        vbt_close_price_mult,
        order_func_nb,
        vbt_open_price_mult.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=pre_group_func_nb,
        pre_group_args=(
            np.array(params["entry"]),
            np.array(params["exit"]),
            np.array(params["delta"]),
            np.array(params["vt"]),
            order_size,
            burnin,
        ),
        pre_segment_func_nb=pre_segment_func_nb,
        pre_segment_args=(transformation, hedge,),
        fill_pos_record=False,
        init_cash=cash,
        cash_sharing=True, 
        group_by=param_columns.names,
        freq=freq,
    )

    res = _analyze_results(pf, interval, rf)
    gc.collect()
    return res

def low_param_simulate_from_order_func(
    close_data:pd.DataFrame, open_data:pd.DataFrame, entry:float,
    exit:float, burnin:int=500, delta:float=1e-5, vt:float=1.0, 
    transformation:str="default", cash:int=100_000, commission:float=0.0008, 
    slippage:float=0.0010, order_size:float=0.10, freq:None or str=None, 
    hedge:str="dollar",
):
    return vbt.Portfolio.from_order_func(
        close_data,
        order_func_nb, 
        open_data.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=pre_group_func_nb, 
        pre_group_args=(
            entry, 
            exit, 
            delta, 
            vt, 
            order_size, 
            burnin,
        ),
        pre_segment_func_nb=pre_segment_func_nb, 
        pre_segment_args=(transformation, hedge,),
        fill_pos_record=False,  # a bit faster
        init_cash=cash,
        cash_sharing=True, 
        group_by=True,
        freq=freq
    )


def simulate_rolling_ols_model(
    close_data:pd.DataFrame, open_data:pd.DataFrame, period:int=60, 
    upper_entry:float=1.25, upper_exit:float=0.75, lower_entry:float=-1.25, 
    lower_exit:float=-0.50, cash:int=100_000, slippage:float=0.0010,
    order_size:float=0.10, freq:str='60T', commission:float=0.0008,
    hedge:str='beta', transformation:str='logret', standard_score:str='sscore',
):
    """Leverage Rolling OLS to estimate Ornstein-Uhlenbeck process. Trade based on 
    standardized spread measures. This strategy is as outlined in Avallaneda et al.
    (2008)."""
    if lower_entry > 0:
        lower_entry = -lower_entry
    if lower_exit > 0:
        lower_exit = -lower_exit

    return vbt.Portfolio.from_order_func(
        close_data,
        order_func_nb, 
        open_data.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=ols_pre_group_func_nb, 
        pre_group_args=(
            period, 
            upper_entry, 
            upper_exit,
            lower_entry, 
            lower_exit, 
            order_size,
        ),
        pre_segment_func_nb=ols_pre_segment_func_nb, 
        pre_segment_args=(
            transformation, hedge, standard_score,
        ),
        fill_pos_record=False,  # a bit faster
        init_cash=cash,
        cash_sharing=True, 
        group_by=True,
        freq=freq,
    )

def simulate_mult_rolling_ols_model(
    close_data:pd.DataFrame, open_data:pd.DataFrame, params:dict, 
    commission:float=0.0008, slippage:float=0.0005, cash:int=100_000, 
    order_size:float=0.10, freq:str="h", standard_score:str='sscore',
    hedge:str='beta', transformation:str='logret', 
):
    """Simultaneously backtest large set of parameter combinations"""
    # Generate multiIndex columns
    param_product = vbt.utils.params.create_param_product(list(params.values()))

    for idx, prod in enumerate(param_product):   # vbt create_param_product creates rounding errors
        prod = [round(num, 11) for num in prod]  # In response, we clean up the numbers
        param_product[idx] = prod                # Then reconstruct the param_product var

    param_tuples = list(zip(*param_product))
    param_columns = pd.MultiIndex.from_tuples(param_tuples, names=params.keys())

    # We need two price columns per param combination
    vbt_close_price_mult = close_data.vbt.tile(len(param_columns), keys=param_columns)
    vbt_open_price_mult = open_data.vbt.tile(len(param_columns), keys=param_columns)

    # Simulate the portfolio and return portfolio object
    pf = vbt.Portfolio.from_order_func(
        vbt_close_price_mult,
        order_func_nb,
        vbt_open_price_mult.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=ols_pre_group_func_nb,
        pre_group_args=(
            np.array(param_product[0]),
            np.array(param_product[1]),
            np.array(param_product[2]),
            np.array(param_product[3]),
            np.array(param_product[4]),
            order_size,
        ),
        pre_segment_func_nb=ols_pre_segment_func_nb,
        pre_segment_args=(
            transformation, hedge, standard_score,
        ),
        fill_pos_record=False,
        init_cash=cash,
        cash_sharing=True,
        group_by=param_columns.names,
        freq=freq,
    )
    return pf

def simulate_batch_rolling_ols_model(
    close_data:pd.DataFrame, open_data:pd.DataFrame, params:dict, 
    commission:float=0.0008, slippage:float=0.0005, hedge:str="beta",
    cash:int=100_000, order_size:float=0.10, interval:str="minutes", 
    freq:str=None, rf:float=0.00, transformation:str='logret', 
    standard_score:str='sscore',
):
    """Test pre-batched parameters on Rolling Ornstein-Uhlenbeck Model from Avallaneda et al. (2008)"""
    # Generate multiIndex columns
    param_tuples = list(zip(*params.values()))
    param_columns = pd.MultiIndex.from_tuples(param_tuples, names=params.keys())

    # We need two price columns per param combination
    vbt_close_price_mult = close_data.vbt.tile(len(param_columns), keys=param_columns)
    vbt_open_price_mult = open_data.vbt.tile(len(param_columns), keys=param_columns)

    # Simulate the portfolio and return portfolio object
    pf = vbt.Portfolio.from_order_func(
        vbt_close_price_mult,
        order_func_nb,
        vbt_open_price_mult.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=ols_pre_group_func_nb,
        pre_group_args=(
            np.array(params["period"]),
            np.array(params["upper_entry"]),
            np.array(params["upper_exit"]),
            np.array(params["lower_entry"]),
            np.array(params["lower_exit"]),
            order_size,
        ),
        pre_segment_func_nb=ols_pre_segment_func_nb,
        pre_segment_args=(
            transformation, hedge, standard_score,
        ),
        fill_pos_record=False,
        init_cash=cash,
        cash_sharing=True, 
        group_by=param_columns.names,
        freq=freq,
    )

    res = _analyze_results(pf, interval, rf=rf)
    gc.collect()
    return res


############################################################################
##                           Copula Models                                ##
############################################################################



############################################################################
##                           Novel Models                                 ##
############################################################################


#############################    Analysis     ###############################

def _analyze_results(pf, interval, burnin=None, rf=None):
    """Analyze output portfolio object for BATCH results"""
    total_return = pf.total_return()    # Extract total return for each param
    wr = extract_wr(pf) # Extract win rate on net long-short trade for each param
    dur = extract_duration(pf, interval) # Extract median trade duration in `interval`
    trades = number_of_trades(pf)
    profit_ratio = calculate_profit_ratio(pf)
    sharpe = custom_sharpe_ratio(pf, burnin, rf)
    sortino = custom_sortino_ratio(pf, burnin, rf)
    res = pd.concat(
        [total_return, wr, dur, trades, profit_ratio, sharpe, sortino], 
        axis=1,
    )
    return res

def transform_seed(seed:np.array, nobs:tuple, transformation:str) -> np.array:
    """Apply price transformations to seed"""
    if not seed.any():
        seed = np.full(nobs, 0, dtype=np.float_)
    elif seed.any() and transformation == 'log':
        seed = np.log(seed)
    elif seed.any() and transformation == 'cumlog':
        seed = cummulative_return_transform(seed)
    elif seed.any() and transformation == 'logret':
        seed = log_return_transform(seed)
    else:
        pass
    return seed
