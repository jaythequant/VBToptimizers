import pandas as pd
import gc
from ._order import simulate_mult_from_order_func


def testParams(
    close_test_sets, open_test_sets, params, commission=0.0008, slippage=0.0010, 
    burnin=5000, cash=100_000, order_size=0.10, freq="m",
) -> pd.Series:

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
        )
        fitness_results.append(df)
        gc.collect()
    
    # Calculate mean results for each param across folds
    cv_results = pd.concat(fitness_results, axis=1) 
    mean_results = cv_results.groupby(by=cv_results.columns, axis=1).mean()

    return mean_results
