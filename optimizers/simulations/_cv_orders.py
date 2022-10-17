import gc
import pandas as pd
from .order import simulate_batch_from_order_func, simulate_batch_from_order_func_low_param
from .order import simulate_from_order_func
from .statistics import score_results, return_results
from .statistics import _weighted_average, _calculate_mse


def trainParams(
    close_train_sets:list, open_train_sets:list, params:dict, commission:float=0.0008, 
    slippage:float=0.0010, burnin:int=500, cash:int=100_000, order_size:float=0.10, 
    freq:None or str=None, hedge:str="dollar", close_validation_sets:None or list=None, 
    open_validation_sets:None or list=None, mode:str="default", model='LQE2', rf=0.05,
) -> pd.DataFrame:
    """Train param batch against cross-validated training (and validation) data.

    Notes
    -----
    For detailed documentation see `optimizers.simulations._order.simulate_batch_from_order_func`

    Parameters
    ----------
    close_train_set : list
    open_train_sets : list
    params : dict
    commission : float, optional
    slippage : float, optional
    burnin : int, optional
    cash : int, optional
    order_size : float, optional
    freq : None or str, optional
    hedge : str, optional
    close_validation_sets : None or list, optional
    open_validation_sets : None or list, optional
    mode : str, optional

    Returns
    -------
    DataFrame
        Returns a dataframe indexed the the given parameter combinations with a 
        series of statistics for evaluation of simulation performance.

    See Also
    --------
    * `optimizers.simulations._order.simulate_batch_from_order_func`
    * `optimizers.simulations.statistics._weighted_average`
    * `optimizers.simulations.statistics._calculate_mse`
    * `vbt.Portfolio`
    """
    fitness_results = []
    validate_results = []
    test_data = zip(close_train_sets, open_train_sets)

    for close_prices, open_prices in test_data:
        if model == 'LQE1':
            df = simulate_batch_from_order_func(
                close_prices, open_prices, params,
                burnin=burnin,
                cash=cash,
                commission=commission,
                slippage=slippage,
                order_size=order_size,
                freq=freq,
                hedge=hedge,
                mode=mode,
                rf=rf
            )
            fitness_results.append(df)
            gc.collect()
        if model == 'LQE2':
            df = simulate_batch_from_order_func_low_param(
                close_prices, open_prices, params,
                burnin=burnin,
                cash=cash,
                commission=commission,
                slippage=slippage,
                order_size=order_size,
                freq=freq,
                hedge=hedge,
                mode=mode,
                rf=rf
            )
            fitness_results.append(df)
            gc.collect()

    # If we pass validation data we will process it as well
    if close_validation_sets and open_validation_sets:
        validate_data = zip(close_validation_sets, open_validation_sets)

        for close_prices, open_prices in validate_data:
            if model == 'LQE1':
                df = simulate_batch_from_order_func(
                    close_prices, open_prices, params,
                    burnin=burnin,
                    cash=cash,
                    commission=commission,
                    slippage=slippage,
                    order_size=order_size,
                    freq=freq,
                    hedge=hedge,
                    rf=rf
                )
                validate_results.append(df)
                gc.collect()
    
    # Calculate mean results for each param across folds
    train_cv_results = pd.concat(fitness_results, axis=1)
    weighted_wr = _weighted_average(train_cv_results)
    mean_results = train_cv_results.groupby(by=train_cv_results.columns, axis=1).mean()
    # median_results = train_cv_results.groupby(by=train_cv_results.columns, axis=1).median()

    if validate_results:
        validate_cv_results = pd.concat(validate_results, axis=1)
        mse, std = _calculate_mse(train_cv_results, validate_cv_results)
        results = pd.concat([mean_results, weighted_wr, mse, std], axis=1)
    else:
        results = pd.concat([mean_results, weighted_wr], axis=1)

    return results


def testParams(
    close_test_sets:list, open_test_sets:list, period:float, upper:float,
    lower:float, exit:float, delta:float=1e-5, vt:float=1.0, burnin:int=500, 
    mode:str="default", cash:int=100_000, commission:float=0.0008, 
    slippage:float=0.0010, order_size:float=0.10, freq:None or str=None, 
    hedge:str="dollar",
):
    """Test unique parameter set against multi-fold test set data

    Notes
    -----
    For detailed documentation see `optimizers.simulations._order.simulate_batch_from_order_func`

    Parameters
    ----------
    close_test_set : list
    open_test_sets : list
    period : float
    upper : float
    lower : float
    exit : float
    delta : float, optional
    vt : float, optional
    burnin : int, optional
    mode : str, optional
    commission : float, optional
    slippage : float, optional
    burnin : int, optional
    cash : int, optional
    order_size : float, optional
    freq : None or str, optional
    hedge : str, optional

    Returns
    -------
    tuple
        Returns a tuple of pandas Series with relevant statistics for 
        evaluation

    See Also
    --------
        `optimizers.simulations._order.simulate_from_order_func`
        `optimizers.simulations.statistics.score_results`
        `optimizers.simulations.statistics.return_results`
        `vbt.Portfolio`
    """
    test_res = []
    test_data = zip(close_test_sets, open_test_sets)

    for close_prices, open_prices in test_data:
        pf = simulate_from_order_func(
            close_prices, open_prices, 
            period=period,
            upper=upper,
            lower=lower,
            exit=exit,
            delta=delta,
            vt=vt,
            burnin=burnin,
            cash=cash, 
            commission=commission, 
            slippage=slippage, 
            order_size=order_size,
            freq=freq,
            hedge=hedge,
            mode=mode,
        )

        test_res.append(pf)
        # For some reason the pf object does not get collected normally
        # As such we need to manual call `gc.collect` to prevent memory bloat
        gc.collect() 

    wr = score_results(test_res)
    total_return = return_results(test_res)
    return wr, total_return
