import logging
import concurrent.futures
import numpy as np
import pandas as pd
from itertools import repeat

from .genetic.operators import init_generate_population
from .genetic.operators import roulette_wheel_selection, crossover, mutation
from .setup.lqe_setup import *
from .setup._cv_orders import testParamsgenetic, testParamsrandom
from .setup.statistics import generate_random_sample
from .genetic.utils import _handle_duplication
from .genetic.utils import _batch_populations
from .utils.cross_validators import vbt_cv_kfold_constructor
from .utils.cross_validators import vbt_cv_timeseries_constructor


def geneticCV(
    opens:pd.DataFrame, closes:pd.DataFrame, params:dict, n_iter:int=100, population:int=100,
    cross_rate:float=1.00, mutation_rate:float=0.05, n_splits:int=5, order_size:float=0.10,
    n_batch_size:int or None=None, max_workers:int or None=None,
    slippage:float=0.0005, cash:int=100_000, freq:str="m",
    rank_method="default", elitism:float or dict=None, mutation_style="random",
    mutation_steps:float or dict=0.10, commission:float=0.0008, 
    n_batches:int or None=None, burnin:int=500, diversity:float or dict=0.00,
    pickle_results:bool=False, hedge="dollar", trade_const=2, cv="timeseries",
) -> pd.DataFrame:
    """Optimize pairs trading strategy via genetic algorithm

    Parameters
    ----------
        opens : pd.DataFrame
            Dataset of open prices per bar for asset pair. Pandas dataframe should
            be indexed to datetime. Be aware that results may be inaccurate if
            index is missing bars, if `freq` parameter is specified differently
            than datatime index frequency, or if datatime index frequency is 
            in consistent.
        closes : pd.DataFrame
            Dataset of close prices per bar for asset pair. This dataframe should
            be specified with an identical index to the opens dataframe. If it is 
            not there will be consistent test results.
        params : dict
        n_iter : int, optional
            Maximum number of iterations to run genetic algorithm for optimization.
        population : int, optional
            Fixed population size per iteration of genetic algorithm. E.g., if
            `population=100`, then 100 parameter combinations will be run per 
            generation for `n_iter` generations or until convergence. 
            Default = 100
        cross_rate : float, optional
            Percentage rate at which each genome will be allowed to produce an 
            offspring with another genome in the population. Default = 1.00
        mutation_rate : float, optional
            Percentage change that a genome will be "mutated". Mutation refers to 
            a randomly selected gene (parameter) within the genome (set of parameters)
            being replaced by a randomly selected *different* gene.
        n_splits : int, optional
            Number of splits to use for cross-validation training. Default = 5. 
            To avoid using cross-validation set `cv=None`.
        n_batch_size : int, optional
        max_workers : int, optional
        min_trades : int
        commission : float
        slippage : float
        cash : int
        order_size : float
        freq : str, optional
            Frequency at which open and close data appears.
        rank_method : str
        burnin : int
            Burnin period before which no rolling statistics are included. 
            Burnin periods are extremely important for the LQE process, but are
            highly subjective to the data used. Please review the following link
            for further information on finding an appropriate burnin period:
        export_results : bool
        n_batches : int or None, optional
            Rather than specifying batch size, we can specify batch count using this
            parameter. See also `n_batch_size` for specifying batch size. Note that
            if `n_batches` and `n_batch_size` as None (defaults), the data will not
            be batches and may overwhelm your machines memory.
        mutation_steps : float
        mutation_style : str

    Returns
    -------
    pd.DataFrame
        DataFrame with multiIndex of parameters and columns showing fitness score and
        supporting statistics for the final generation in the GA process. 
    """
    if cv == "kfold":
        close_train_dfs, _ = vbt_cv_kfold_constructor(closes, n_splits=n_splits)
        open_train_dfs, _ = vbt_cv_kfold_constructor(opens, n_splits=n_splits)
    if cv == "timeseries":
        close_train_dfs, _ = vbt_cv_timeseries_constructor(closes, n_splits=n_splits)
        open_train_dfs, _ = vbt_cv_timeseries_constructor(opens, n_splits=n_splits)
    
    # Generate initial population
    generation = init_generate_population(params, population=population)

    for i in range(n_iter):

        if isinstance(diversity, dict):
            for stage, const in diversity.items():
                if i == stage:
                    diversity_const = const
        else:
            diversity_const = diversity
        
        if isinstance(elitism, dict):
            for stage, const in elitism.items():
                if i == stage:
                    elitism_const = const
        else:
            elitism_const = elitism

        # Batch the population for parallelization and memory concerns
        batches = _batch_populations(
            generation, n_batch_size=n_batch_size, n_batches=n_batches
        )

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
                repeat(hedge),
            ):
                results.append(result)

        df = pd.concat(results)

        adjustor = (1 - (1 / df["trade_count"])) ** trade_const
        df["fitness"] = df["Weighted Average"] * adjustor
        
        logging.info(f"Iteration {i} completed")
        logging.info('\n\t'+ df.sort_values("fitness", ascending=False).head(10).to_string().replace('\n', '\n\t'))

        # Use roulette wheel, random crossover, and mutation to produce next generation
        g = roulette_wheel_selection(
            df, params.keys(),
            population=population,
            rank_method=rank_method,
            rank_space_constant=elitism_const,
            diversity_constant=diversity_const,
        )
        generation = crossover(g, cross_rate=cross_rate)
        generation = mutation(
            generation, params,
            mutation_rate=mutation_rate,
            style=mutation_style,
            step_size=mutation_steps,
        )
        generation = _handle_duplication(
            generation, params, handler=mutation_style, step_size=mutation_steps
        )

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

