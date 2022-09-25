import logging
import concurrent.futures
import numpy as np
import pandas as pd
from itertools import repeat
from .genetic.operators import init_generate_population
from .genetic.operators import roulette_wheel_selection, crossover, mutation
from .simulations._cv_orders import trainParams, testParams
from .simulations.statistics import generate_random_sample
from .genetic.utils import _handle_duplication
from .genetic.utils import _batch_populations
from .utils.cross_validators import vbt_cv_kfold_constructor
from .utils.cross_validators import vbt_cv_sliding_constructor
from .utils.cross_validators import vbt_cv_timeseries_constructor

logger = logging.getLogger(__name__)


def geneticCV(
    opens:pd.DataFrame, closes:pd.DataFrame, params:dict, n_iter:int=100, population:int=100,
    cross_rate:float=1.00, mutation_rate:float=0.05, n_splits:int=5, order_size:float=0.10,
    n_batch_size:int or None=None, max_workers:int or None=None, model='LQE2',
    slippage:float=0.0005, cash:int=100_000, freq:str="h",
    rank_method="default", elitism:float or dict=None, mutation_style="random",
    mutation_steps:float or dict=0.10, commission:float=0.0008, 
    n_batches:int or None=None, burnin:int=500, diversity:float or dict=0.00,
    pickle_results:bool=False, hedge="dollar", trade_const:float=1.0, cv:str="timeseries",
    mode="default", validation_set:None or float=None, ret_const:float=1.0,
    sr_const:float=1.0, wr_const:float=1.0, total_return_min:float=0.00, 
    duration_cap:int=1440, dur_const:float=0.50, trade_floor:int=35
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
            be specified with an identical index to the opens dataframe. Results will
            be corrupted if indices differ in any way.
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
            Percentage chance that a genome will be "mutated". Mutation refers to 
            a randomly selected gene (parameter) within the genome (set of parameters)
            being replaced by a randomly selected *different* gene.
        n_splits : int, optional
            Number of splits to use for cross-validation training. Default = 5. 
            To avoid using cross-validation set `cv=None`.
        n_batch_size : int, optional
        max_workers : None or int, optional
            Max allowable workers for multiprocessing. If `max_workers=None`, a variable
            number of workers will be created optimized to the amount of compute available.
            See `concurrent.futures.ProcessPoolExecutor` documentation for more.
        commission : float, optional
            Set fixed percentage commission charged per trade placed during simulation.
            Default value is `0.0008`
        slippage : float, optional
            Set fixed percentage slippage applied to each trade placed during simulation.
            Be aware that slippage will always make the trade less favorable and will apply
            negative slippage to short trades and positive slippage to long trades ensuring
            that slippage always worsens the return, never aids it. Default = `0.0005`
        cash : int, optional
            Set starting cash for portfolio object. Default = `100_000`
        order_size : float, optional
            Set in percentage terms the amount of the portfolio to allocate to a single trade.
            Note that as this is a long short strategy `order_size` * `portfolio.value()`
            units will be allocated to **both** sides of the trade.
        freq : str, optional
            Frequency at which open and close data appears. If `freq=None`, `vectorBT` will 
            attempt to surmise the appropriate frequency. Default = `None`
        rank_method : str
        burnin : int
            Burnin period before which no rolling statistics are included. 
            Burnin periods are extremely important for the LQE process, but are
            highly subjective to the data used. Please review the following link
            for further information on finding an appropriate burnin period:
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
        close_train_dfs, close_validate_dfs = vbt_cv_kfold_constructor(closes, n_splits=n_splits)
        open_train_dfs, open_validate_dfs = vbt_cv_kfold_constructor(opens, n_splits=n_splits)
    if cv == "timeseries":
        close_train_dfs, close_validate_dfs = vbt_cv_timeseries_constructor(closes, n_splits=n_splits)
        open_train_dfs, open_validate_dfs = vbt_cv_timeseries_constructor(opens, n_splits=n_splits)
    if cv == "sliding":
        if validation_set:
            close_train_dfs, close_validate_dfs = vbt_cv_sliding_constructor(
                closes, n_splits=n_splits, set_lens=(validation_set,)
            )
            open_train_dfs, open_validate_dfs = vbt_cv_sliding_constructor(
                opens, n_splits=n_splits, set_lens=(validation_set,)
            )
        else:
            close_train_dfs = vbt_cv_sliding_constructor(closes, n_splits=n_splits)
            open_train_dfs = vbt_cv_sliding_constructor(opens, n_splits=n_splits)
            close_validate_dfs = None
            open_validate_dfs = None
    
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
                trainParams, 
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
                repeat(open_validate_dfs),
                repeat(close_validate_dfs),
                repeat(mode),
                repeat(model),
            ):
                results.append(result)

        df = pd.concat(results)

        df["sharpe_ratio"] = np.where(df["sharpe_ratio"] == np.inf, 0, df["sharpe_ratio"])
        df["trade_count"] = np.where(df["trade_count"] == 0, 1., df["trade_count"]) # Prevents np.log error
        df["fitness"] = (
            (sr_const * df["sharpe_ratio"]/5) +
            (wr_const * df["Weighted Average"]) +
            (trade_const * np.log(df["trade_count"]/2)) # Divided by 2 should eventually be removed
        )
        df["fitness"] = np.where(
            df["total_return"] < total_return_min,
            df["fitness"] * ret_const,
            df["fitness"],
        ) # Punish negative returns
        df["fitness"] = np.where(
            df["duration"] > duration_cap,
            df["fitness"] * (dur_const/df["duration"]),
            df["fitness"]
        ) # Punish high duration trades
        df["fitness"] = np.where(
            df["trade_count"] < trade_floor,
            df["fitness"] * (df["trade_count"]/trade_floor),
            df["fitness"]
        ) # Punish low trade counts
        df["fitness"] = np.where(df.fitness < 0, 0, df.fitness) # Ensure that fitness cannot be negative
        
        logging.info(f"Iteration {i} completed")
        logging.info('\n\t'+ df.sort_values("fitness", ascending=False).head(15).to_string().replace('\n', '\n\t'))

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
        for result in executor.map(testParams, repeat(close_train_dfs), repeat(open_train_dfs), sample_set):
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

