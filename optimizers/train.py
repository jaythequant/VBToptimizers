import logging
import concurrent.futures
import numpy as np
import pandas as pd
from itertools import repeat

from .genetic.setup import init_generate_population
from .genetic.setup import roulette_wheel_selection, crossover, mutation
from .setup.lqe_setup import *
from .setup._cv_orders import testParamsgenetic, testParamsrandom
from .setup.statistics import generate_random_sample
from .genetic.utils import _handle_duplication
from .genetic.utils import _batch_populations
from .utils.cross_validators import vbt_cv_kfold_constructor


def geneticCV(
    opens:pd.Series, closes:pd.Series, params:dict, n_iter:int=100, population:int=100,
    cross_rate:float=1.00, mutation_rate:float=0.05, handler:str="mutate", n_splits:int=5,
    n_batch_size:int=100, max_workers:int=4, min_trades:int=5, commission:float=0.0008,
    slippage:float=0.0005, cash:int=100_000, order_size:float=0.10, freq:str="m",
    rank_method="default", rank_space_constant=None, burnin:int=500, export_results=True,
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
        rank_method : str
        rank_space_constant : float or None
        burnin : int
        export_results : bool

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

    df = None

    for i in range(n_iter):
        # Batch the population for parallelization and memory concerns
        batches = _batch_populations(generation, n_batch_size=n_batch_size)

        results = []

        # Execute our cross validation function concurrently on `max_worker` processors
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(
                testParamsgenetic, 
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

        logging.info(f"Iteration {i} completed")
        logging.info(df.sort_values(by="fitness", ascending=False))
        
        if export_results:
            df.to_csv(f"results_generation_{i}.csv")

        # Use roulette wheel, random crossover, and mutation to produce next generation
        g = roulette_wheel_selection(
            df, params.keys(),
            population=population,
            rank_method=rank_method,
            rank_space_constant=rank_space_constant,
        )
        generation = crossover(g, cross_rate=cross_rate)
        generation = mutation(generation, params, mutation_rate=mutation_rate)
        generation = _handle_duplication(generation, params, handler=handler)

        # Measure and report some statistics
        most_fit = df.fitness.idxmax()
        highest_wr = df.fitness.max()
        average_wr = df.fitness.mean()
        s = highest_wr - average_wr
        logging.info(f"{most_fit} ---> {highest_wr:.4f}")
        logging.info(f"Spread: {highest_wr:.4f} - {average_wr:.4f} = {s:.4f}")

    return df


def randomSearch(
    close_data_set:pd.DataFrame, open_data_set:pd.DataFrame, 
    workers:int=4, n_iter:int=1000, n_splits:int=5
):
    """Parallelized random search algorithm using kfold cross-validation and vectorBT framework

    Random Search CV algorithm implementating vectorized backtesting via VectorBT, kfold cross
    validation via sklearn's KFold class, and randomized search of a provided search space via
    `numpy.random` functions. Search process is fully parallelized using `concurrent.futures`.
    
    Parameters
    ----------
        close_data_set : pd.DataFrame
            pandas DataFrame formatting with a datetime index and two closing price columns for the
            X and y asset being fed to the model. 
        open_data_set : pd.DataFrame
            Identically formatted pandas DataFrame to `close_data_set` (e.g., datetime index with X 
            and y price columns). Price data should be open prices and datetime index must match
            `close_data_set` datetime index exactly.
        workers : int
            Max number of workers to allow for parallelization of random search process. See 
            `concurrent.futures.ProcessPoolExecutor` documentation for more details
        n_iter : int
            Number of iterations to perform (i.e., random samples to gather and test) for population
            of possible samples
        n_splits : int
            Number of folds to generate during kfold generation process. For more details on kfolds 
            and the underlying crossvalidation process used see `sklearn.model_selection.KFold` 
            documentation.

    Returns
    -------
    tuple
        Returns a tuple containing (in this order):
        * The best parameter combination other of sample tested
        * The best parameters win rate
        * The best parameters total return
    """

    close_train_dfs, _ = vbt_cv_kfold_constructor(close_data_set, n_splits=n_splits)
    open_train_dfs, _ = vbt_cv_kfold_constructor(open_data_set, n_splits=n_splits)

    sample_set = generate_random_sample(n_iter=n_iter) # Random sample set of site n_iter created

    # Initialize an empty dataframe to store records 
    df = pd.DataFrame(columns=[
        "period", "upper", "lower", "exit", 
        "delta","vt", "wr", "ret"
        ]
    ) 

    best_comb = None
    best_wr = None
    best_ret = None

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(testParamsrandom, repeat(close_train_dfs), repeat(open_train_dfs), sample_set):
            sample, wr, ret = result # Unpack out result into params, win rate, and total return
            if not best_comb:
                best_comb = sample
                best_wr = wr
                best_ret = ret
                logging.info("Initial params:", best_comb)
                logging.info(f"Score: {wr:.4f}")
                logging.info(f"Return: {ret}")
            if wr > best_wr:
                best_comb = sample
                best_wr = wr
                best_ret = ret
                logging.info("New best param:", best_comb)
                logging.info(f"Score: {wr:.4f}")
                logging.info(f"Return: {ret}")
            stats = {"wr": wr, "ret": ret}
            stats.update(sample)
            # Wrap a dictionary with param details + win rate and total return info in a dataframe
            r = pd.DataFrame.from_dict(
                stats, orient="index"
            ).T 
            # Add to dataframe containing records of all params tested
            df = pd.concat([df, r]) 

    df.to_csv("res.csv", index=False) # Export the results to a CSV for review later

    return best_comb, best_wr, best_ret

