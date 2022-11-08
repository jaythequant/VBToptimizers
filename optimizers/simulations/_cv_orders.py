import gc
import pandas as pd
import numpy as np
from .order import simulate_lqe_model
from .order import (
    simulate_batch_lqe_model, 
    simulate_batch_from_order_func_low_param,
)
from .strategies.components.statistics import (
    score_results, return_results, _weighted_average
)

def pairs_cross_validator(
    close_train_sets:list, open_train_sets:list, params:dict, commission:float=0.0008, 
    slippage:float=0.0010, burnin:int=500, cash:int=100_000, order_size:float=0.10, 
    freq:str=None, hedge:str="dollar", transformation:str="default", model='LQE2', 
    rf=0.00, standard_score='zscore', seed=False,
) -> pd.DataFrame:
    """Train param batch against cross-validated training (and validation) data.

    Notes
    -----
    For detailed documentation see `optimizers.simulations.order.simulate_batch_from_order_func`

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
    transformation : str, optional
    seed_filter : str, optional

    Returns
    -------
    DataFrame
        Return a dataframe indexed to parameter combinations with a 
        series of statistics for evaluating simulation performance.

    See Also
    --------
    * `optimizers.simulations.order.simulate_batch_from_order_func`
    * `optimizers.simulations.statistics._weighted_average`
    * `optimizers.simulations.statistics._calculate_mse`
    * `vbt.Portfolio`
    """
    fitness_results = []
    test_data = zip(close_train_sets, open_train_sets)

    for idx, (close_prices, open_prices) in enumerate(test_data):
        if model == 'LQE':
            if (seed and idx == 0) or not seed:
                seed_set = np.array([])
            elif seed and idx != 0:
                seed_set = pd.concat(close_train_sets[:idx]).values
            df = simulate_batch_lqe_model(
                close_prices, open_prices, params,
                burnin=burnin,
                cash=cash,
                commission=commission,
                slippage=slippage,
                order_size=order_size,
                freq=freq,
                hedge=hedge,
                transformation=transformation,
                rf=rf,
                standard_score=standard_score,
                seed=seed_set
            )
            fitness_results.append(df)
            gc.collect()
        elif model == 'LQE2':
            df = simulate_batch_from_order_func_low_param(
                close_prices, open_prices, params,
                burnin=burnin,
                cash=cash,
                commission=commission,
                slippage=slippage,
                order_size=order_size,
                freq=freq,
                hedge=hedge,
                model=transformation,
                rf=rf,
            )
            fitness_results.append(df)
            gc.collect()
        else:
            raise ValueError(f'No {model} model found in simulations')
    
    # Calculate mean results for each param across folds
    train_cv_results = pd.concat(fitness_results, axis=1)
    train_cv_results = train_cv_results.fillna(0)
    weighted_wr = _weighted_average(train_cv_results)
    mean_results = train_cv_results.groupby(by=train_cv_results.columns, axis=1).mean()

    return pd.concat([mean_results, weighted_wr], axis=1)


def testParams(
    close_test_sets:list, open_test_sets:list, period:float, upper:float,
    lower:float, exit:float, delta:float=1e-5, vt:float=1.0, burnin:int=500, 
    transformation:str="default", cash:int=100_000, commission:float=0.0008, 
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
    transformation : str, optional
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
        pf = simulate_lqe_model(
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
            transformation=transformation,
        )

        test_res.append(pf)
        # For some reason the pf object does not get collected normally
        # As such we need to manual call `gc.collect` to prevent memory bloat
        gc.collect() 

    wr = score_results(test_res)
    total_return = return_results(test_res)
    return wr, total_return
