import vectorbt as vbt
import pandas as pd
import numpy as np
import gc
import logging
import math
import concurrent.futures
from itertools import repeat

from setup.lqe_setup import *    # Import LQE model setup
from utils.cross_validators import vbt_cv_kfold_constructor
from utils.statistics import extract_duration, extract_wr

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

def simulate_mult_from_order_func(
    close_prices, open_prices, batch, commission=0.0008, 
    slippage=0.0010, mode="Kalman", cash=100_000, 
    order_size=0.10, burnin=500, freq="m", interval="minutes",
):
    """Execute gridsearch backtesting with specified set of parameters"""

    # Generate multiIndex columns
    param_tuples = list(zip(*batch))
    param_columns = pd.MultiIndex.from_tuples(param_tuples, names=batch.keys())

    # We need two price columns per param combination
    vbt_close_price_mult = close_prices.vbt.tile(len(param_columns), keys=param_columns)
    vbt_open_price_mult = open_prices.vbt.tile(len(param_columns), keys=param_columns)

    # Simulate the portfolio and return portfolio object
    pf = vbt.Portfolio.from_order_func(
        vbt_close_price_mult,
        order_func_nb, 
        vbt_open_price_mult.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=pre_group_func_nb,
        pre_group_args=(
            np.array(batch[0]),
            np.array(batch[1]), 
            np.array(batch[2]), 
            np.array(batch[3]), 
            np.array(batch[4]), 
            np.array(batch[5]), 
            order_size,
            burnin,
        ),
        pre_segment_func_nb=pre_segment_func_nb,
        pre_segment_args=(mode,),
        fill_pos_record=False,
        init_cash=cash,
        cash_sharing=True, 
        group_by=param_columns.names,
        freq=freq,
    )

    # Append results of each param comb to CSV file
    total_return = pf.total_return()    # Extract total return for each param
    wr = extract_wr(pf) # Extract win rate on net long-short trade for each param
    dur = extract_duration(pf, interval) # Extract median trade duration in hours
    # Append batch results to datafrane for later analysis
    df = pd.concat([total_return, wr, dur], axis=1)
    # print(df)
    # Collect and discard garbage
    # gc.collect()
    return df


def testBatch(
    close_training_sets, open_training_sets, param_batch, 
    n_batch_splits=10, interval="days",
) -> None:
    """Train machine learning models using gridsearch algorithm and kfold cross-validation
    
    Parameters
    ----------
    """
    train_data = zip(close_training_sets, open_training_sets)

    for idx, (close_prices, open_prices) in enumerate(train_data):

        simulate_mult_from_order_func(
            close_prices, open_prices, param_batch, 
            id=idx, interval=interval,
        )


def gridsearchCV(
    close_prices:pd.DataFrame, open_prices:pd.DataFrame, 
    params:dict, mode:str="Kalman", cash:int=100_000, burnin:int=5000, 
    commission:float=0.0008, slippage:float=0.0010, order_size:float=0.10, 
    freq:str="d", n_batch_splits:int=10, id:int=None, interval:str="days",
) -> None:
    """Simulate multiple parameter combinations using `Portfolio.from_order_func`.

    Parameters
    ----------

    close_prices : pd.Series or pd.DataFrame
        Pandas dataframe or Series with datetime index. For best accuracy this series 
        represents close prices.
    :param open_prices: Pandas dataframe or Series with datetime index
        For best accuracy this series represents open prices. 
    """
    # Create a cartesian product for param values
    param_product = vbt.utils.params.create_param_product(params.values()) 

    for idx, prod in enumerate(param_product):  # vbt create_param_product creates rounding errors
        prod = [round(num, 10) for num in prod]  # In response, we clean up the numbers
        param_product[idx] = prod               # Then reconstruct the param_product var

    batches = np.array_split(np.array(param_product), n_batch_splits, axis=1) # convert param_product to n batches

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(testBatch, repeat(close_prices), repeat(open_prices), batches):
            print(result)

    ## Ignore these lines ##
    # repeat(params),
    # repeat(commission),
    # repeat(slippage),
    # repeat(mode),
    # repeat(cash),
    # repeat(order_size),
    # repeat(burnin),
    # repeat(freq),
    # repeat(interval),
    

def parse_batch_results():
    """Parse dataframes produced by testBatch functions"""
    pass


if __name__ == "__main__":

    logging.info("Initializing gridsearch . . .")

    # Initial data acquisition and processing
    sql_close_prices = pd.read_csv("vecbt/data/crypto_close_data.csv", index_col="time")
    sql_open_prices = pd.read_csv("vecbt/data/crypto_open_data.csv", index_col="time")

    sql_close_prices.index = pd.to_datetime(sql_close_prices.index)
    sql_open_prices.index = pd.to_datetime(sql_open_prices.index)

    # Split data into train/test using kfold constructor; discard test data using _
    close_train_dfs, _ = vbt_cv_kfold_constructor(sql_close_prices)
    open_train_dfs, _ = vbt_cv_kfold_constructor(sql_open_prices)

    # Establish numpy arrays containing params
    periods = np.arange(50_000, 50_500, 500, dtype=np.float_)
    uppers = np.arange(3.0, 3.5, 0.5, dtype=np.float_)
    lowers = -1 * np.arange(3.0, 3.5, 0.5, dtype=np.float_)
    exits = np.arange(1.0, 1.1, 0.2, dtype=np.float_)
    vt = np.arange(0.1, 1.1, 0.1, dtype=np.float_)
    delta = 0.1 ** np.arange(10, dtype=np.float_)

    # Wrap params into dictionary for consumption by portfolio simulator
    d_params = {
        "periods": periods, 
        "uppers": uppers, 
        "lowers": lowers, 
        "exits": exits, 
        "vt": vt,
        "delta": delta,
    }

    N_BATCHES = 10  # Number batch splits

    # Calculate the number of param combs needed
    res = 1
    for key, vals in d_params.items():
        res = res * len(vals)
    # Log relevant information on batch, combinations, and training sets
    logging.info(f"Total number of combinations generated: {res}")
    logging.info(f"Number of batches required: {math.ceil(res/N_BATCHES)}")
    logging.info(f"Count of training sets: {len(close_train_dfs)}")

    testBatch(
        close_train_dfs, 
        open_train_dfs, 
        params=d_params,
        order_size=0.10, 
        freq="M", 
        n_batch_splits=N_BATCHES,
        burnin=500,
        interval="minutes"
    )

    logging.info("Finished training model")
