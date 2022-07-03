import pandas as pd
import gc
from ._order import simulate_mult_from_order_func
from ._order import simulate_from_order_func
from .statistics import score_results, return_results
from .statistics import _weighted_average, _calculate_mse


def validateParamsgenetic(
    close_train_sets, open_train_sets, params, commission=0.0008, slippage=0.0010, 
    burnin=5000, cash=100_000, order_size=0.10, freq="m", hedge="dollar",
    close_validation_sets=None, open_validation_sets=None,
) -> pd.DataFrame:

    fitness_results = []
    validate_results = []
    test_data = zip(close_train_sets, open_train_sets)

    for close_prices, open_prices in test_data:
        df = simulate_mult_from_order_func(
            close_prices, open_prices, params,
            burnin=burnin,
            cash=cash,
            commission=commission,
            slippage=slippage,
            order_size=order_size,
            freq=freq,
            hedge=hedge,
        )
        fitness_results.append(df)
        gc.collect()

    # If we pass validation data we will process it as well
    if close_validation_sets and open_validation_sets:
        validate_data = zip(close_validation_sets, open_validation_sets)

        for close_prices, open_prices in validate_data:
            df = simulate_mult_from_order_func(
                close_prices, open_prices, params,
                burnin=burnin,
                cash=cash,
                commission=commission,
                slippage=slippage,
                order_size=order_size,
                freq=freq,
                hedge=hedge,
            )
            validate_results.append(df)
            gc.collect()
    
    # Calculate mean results for each param across folds
    train_cv_results = pd.concat(fitness_results, axis=1)
    weighted_wr = _weighted_average(train_cv_results)
    mean_results = train_cv_results.groupby(by=train_cv_results.columns, axis=1).mean()

    if validate_results:
        validate_cv_results = pd.concat(validate_results, axis=1)
        mse, std = _calculate_mse(train_cv_results, validate_cv_results)
        results = pd.concat([mean_results, weighted_wr, mse, std], axis=1)
    else:
        results = pd.concat([mean_results, weighted_wr], axis=1)

    return results


def testParamsgenetic(
    close_test_sets, open_test_sets, params, commission=0.0008, slippage=0.0010, 
    burnin=5000, cash=100_000, order_size=0.10, freq="m", hedge="dollar",
) -> pd.DataFrame:

    fitness_results = []
    test_data = zip(close_test_sets, open_test_sets)

    for close_prices, open_prices in test_data:
        df = simulate_mult_from_order_func(
            close_prices, open_prices, params,
            burnin=burnin,
            cash=cash,
            commission=commission,
            slippage=slippage,
            order_size=order_size,
            freq=freq,
            hedge=hedge,
        )
        fitness_results.append(df)
        gc.collect()
    
    # Calculate mean results for each param across folds
    cv_results = pd.concat(fitness_results, axis=1)
    weighted_wr = _weighted_average(cv_results)
    mean_results = cv_results.groupby(by=cv_results.columns, axis=1).mean()
    results = pd.concat([mean_results, weighted_wr], axis=1)

    return results


def testParamsrandom(
    close_test_sets, open_test_sets, params, commission=0.0008, slippage=0.0010, 
    burnin=5000, cash=100_000, order_size=0.10, freq="m", hedge="dollar",
):

    test_res = []
    test_data = zip(close_test_sets, open_test_sets)

    for close_prices, open_prices in test_data:
        pf = simulate_from_order_func(
            close_prices, open_prices, 
            period=params["period"],
            upper=params["upper"],
            lower=params["lower"],
            exits=params["exit"],
            delta=params["delta"],
            vt=params["vt"],
            burnin=burnin,
            cash=cash, 
            commission=commission, 
            slippage=slippage, 
            order_size=order_size,
            freq=freq,
            hedge=hedge,
        )

        test_res.append(pf)
        # For some reason the pf object does not get collected normally
        # As such we need to manual call `gc.collect` to prevent memory bloat
        gc.collect() 

    wr = score_results(test_res)
    total_return = return_results(test_res)
    return (params, wr, total_return)
