import numpy as np
import vectorbt as vbt
import logging
import pandas as pd
import numpy as np
import concurrent.futures
from itertools import repeat

from .genetic.setup import init_generate_population
from .genetic.setup import roulette_wheel_selection, crossover, mutation
from .setup.lqe_setup import *
from .setup._cv_orders import testParams
from .genetic.utils import _handle_duplication
from .genetic.utils import _batch_populations
from .utils.cross_validators import vbt_cv_kfold_constructor


def geneticCV(
    opens:pd.Series, closes:pd.Series, params:dict, n_iter:int=100, population:int=100, 
    cross_rate:float=1.00, mutation_rate:float=0.05, handler:str="mutate", n_splits:int=5, 
    n_batch_size:int=100, max_workers:int=4, min_trades:int=5, commission:float=0.0008, 
    slippage:float=0.0005, cash:int=100_000, order_size:float=0.10, freq:str="m", 
    squared_probabilities=False, burnin:int=500,
) -> pd.DataFrame:
    """Execute genetic algorithm `n_iter` times on data set or until convergence fitnesses
    
    Parameters
    ----------
        opens : pd.Series
        closes : pd.Series
        params : dict
        n_iter : int
        population : int
        cross_rate : float
        mutation_rate : float
        handler : str
        n_splits : int
        n_batch_size : int
        max_workers : int
        min_trades : int
        commission : float
        slippage : float
        cash : int
        order_size : float
        freq : str
        squared_probabilities : bool
        burnin : int

    Returns
    -------
    pd.DataFrame
        DataFrame with multiIndex of parameters and columns showing fitness score and
        supporting statistics for the final generation in the GA process. 
    """

    # Use kfold cross-validator to generate `n_split` folds
    close_train_dfs, _ = vbt_cv_kfold_constructor(closes, n_splits=n_splits)
    open_train_dfs, _ = vbt_cv_kfold_constructor(opens, n_splits=n_splits)
    
    # Generate initial population
    generation = init_generate_population(params, population=population)

    most_recent_results = None

    for i in range(n_iter):
        # Batch the population for parallelization and memory concerns
        batches = _batch_populations(generation, n_batch_size=n_batch_size)

        results = []

        # Execute our cross validation function concurrently on `max_worker` processors
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(
                testParams, 
                repeat(close_train_dfs), 
                repeat(open_train_dfs), 
                batches,
                repeat(commission),
                repeat(slippage),
                repeat(burnin),
                repeat(cash),
                repeat(order_size),
                repeat(freq),
            ):
                results.append(result)

        df = pd.concat(results)
        df.rename(columns={"Win Rate": "fitness"}, inplace=True)

        # TEMPORARY MAY CHANGE LATER # 
        # If the parameter combination executed less than `min_trades`, set fitness=0 for that param
        df["fitness"] = np.where(df["trade_count"] < min_trades, 0, df["fitness"])

        # Use roulette wheel, random crossover, and mutation to produce next generation
        g = roulette_wheel_selection(
            df, params.keys(), 
            population=population, 
            squared_prob=squared_probabilities
        )
        generation = crossover(g, cross_rate=cross_rate)
        generation = mutation(generation, params, mutation_rate=mutation_rate)
        generation = _handle_duplication(generation, params, handler=handler)

        # Measure and report some statistics
        most_recent_results = df
        most_fit = most_recent_results.fitness.idxmax()
        highest_wr = most_recent_results.fitness.max()
        average_wr = most_recent_results.fitness.mean()
        s = highest_wr - average_wr
        print(most_recent_results.sort_values("fitness", ascending=True))
        logging.info(f"Iteration {i} completed")
        logging.info(f"{most_fit} ---> {highest_wr:.4f}")
        logging.info(f"Spread: {highest_wr:.4f} - {average_wr:.4f} = {s:.4f}")
    
    return most_recent_results
