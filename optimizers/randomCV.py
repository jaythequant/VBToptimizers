import pandas as pd
import concurrent.futures
from itertools import repeat
import logging

from setup.order_funcs import simulate_from_order_func
from utils.cross_validators import vbt_cv_kfold_constructor
from utils.statistics import generate_random_sample
from utils.statistics import score_results, return_results

# Logging config
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
fmt = "%(asctime)s [%(levelname)s] %(module)s :: %(message)s"
logging.basicConfig(
    format=fmt,
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(filename="train.log"),
        stream_handler,
    ]
)

logger = logging.getLogger(__name__)

def testParams(
    close_test_sets, open_test_sets, params, commission=0.0008, slippage=0.0010, 
    burnin=5000, cash=100_000, order_size=0.10, freq="m",
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
        )

        test_res.append(pf)

    wr = score_results(test_res)
    total_return = return_results(test_res)
    return (params, wr, total_return)


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

    close_train_dfs, close_test_data = vbt_cv_kfold_constructor(close_data_set, n_splits=n_splits)
    open_train_dfs, open_test_data = vbt_cv_kfold_constructor(open_data_set, n_splits=n_splits)

    # Extract the testing datasets from program and remove from memory
    close_test_data.to_csv("close_test_data.csv")
    open_test_data.to_csv("open_test_data.csv")
    del close_test_data
    del open_test_data

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


if __name__ == "__main__":

    logging.info("Initializing training session . . . ")

    closes = pd.read_csv("optimizers/data/crypto_close_data.csv", index_col="time")
    opens = pd.read_csv("optimizers/data/crypto_open_data.csv", index_col="time")

    best_comb, wr, ret = randomSearch(closes, opens, workers=5, n_iter=10_000, n_splits=5)

    logging.info("Tests complete:")
    logging.info("Optimized parameters:", best_comb)
    logging.info(f"Score: {wr:.4f}")
    logging.info(f"Return: {ret}")
    logging.info("Shutting down training session")
