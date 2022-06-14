import vectorbt as vbt
import pandas as pd
import numpy as np
import gc
import logging
import math
from progress.bar import Bar

from setup.lqe_setup import *    # Import LQE model setup
from utils.cross_validators import vbt_cv_kfold_constructor
from utils.statistics import extract_duration, extract_wr

#########################################################
## IMPLEMENTATION OF GRIDSEARCH WITHOUT MULTIPROCESSING##
#########################################################

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

def sim_order(
    close_prices, open_prices, params, commission, slippage, mode, 
    cash, order_size, burnin, freq, interval, id, bar, batch
):
    # Generate multiIndex columns
    param_tuples = list(zip(*batch))
    param_columns = pd.MultiIndex.from_tuples(param_tuples, names=params.keys())

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
    # Append batch results to CSV for later analysis
    pd.concat([total_return, wr, dur], axis=1).to_csv(f"fold_{id}.csv", mode="a", header=False)
    # Collect and discard garbage
    gc.collect()
    bar.next()


def simulate_mult_from_order_func_batcher(
    close_prices:np.array, open_prices:np.array, params:dict, mode:str="Kalman", cash:int=100_000, 
    burnin:int=5000, commission:float=0.0008, slippage:float=0.0010, order_size:float=0.10, 
    freq:str="d", n_batch_splits:int=10, id=None, interval="days",
) -> None:
    """Simulate multiple parameter combinations using `Portfolio.from_order_func`.

    Parameters
    ----------

    :params close_prices: Pandas dataframe or Series with datetime index
        For best accuracy this series represents close prices.
    :param open_prices: Pandas dataframe or Series with datetime index
        For best accuracy this series represents open prices. 
    """
    # Create a cartesian product for param values
    param_product = vbt.utils.params.create_param_product(params.values()) 

    for idx, prod in enumerate(param_product):  # vbt create_param_product creates rounding errors
        prod = [round(num, 10) for num in prod]  # In response, we clean up the numbers
        param_product[idx] = prod               # Then reconstruct the param_product var

    batches = np.array_split(np.array(param_product), n_batch_splits, axis=1) # convert param_product to n batches

    with Bar("Processing", max=len(batches), suffix='%(percent)d%%') as bar:
        for batch in batches:
            # Generate multiIndex columns
            param_tuples = list(zip(*batch))
            param_columns = pd.MultiIndex.from_tuples(param_tuples, names=params.keys())

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
                    np.array(batch[0]), # Periods
                    np.array(batch[1]), # Uppers
                    np.array(batch[2]), # Lowers
                    np.array(batch[3]), # Exits
                    np.array(batch[4]), # Delta
                    np.array(batch[5]), # Vt
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
            # Append batch results to CSV for later analysis
            pd.concat([total_return, wr, dur], axis=1).to_csv(f"fold_{id}.csv", mode="a", header=False)
            # Collect and discard garbage
            gc.collect()
            # Progress the progress bar
            bar.next()
    bar.finish()

def gridsearchCV(
    close_training_sets, open_training_sets, params, 
    mode="Kalman", cash=100_000, burnin=5000, 
    commission=0.0008, slippage=0.0010, order_size=0.10, 
    freq="d", n_batch_splits=10, interval="days",
) -> None:
    """Training function for the purpose of gridsearch utilizing cross validation data"""

    train_data = zip(close_training_sets, open_training_sets)

    for idx, (close_prices, open_prices) in enumerate(train_data):

        simulate_mult_from_order_func_batcher(
            close_prices, open_prices, params=params, mode=mode,
            cash=cash, burnin=burnin, commission=commission, 
            slippage=slippage, order_size=order_size, freq=freq,
            n_batch_splits=n_batch_splits, id=idx, interval=interval,
        )


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
    exits = np.arange(1.5, 1.6, 0.2, dtype=np.float_)
    delta = 0.1 ** np.arange(1, 10, 1, dtype=np.float_)
    vt = np.arange(0.1, 1.1, 0.1, dtype=np.float_)

    # Wrap params into dictionary for consumption by portfolio simulator
    d_params = {
        "periods": periods, 
        "uppers": uppers, 
        "lowers": lowers, 
        "exits": exits, 
        "delta": delta,
        "vt": vt,
    }

    N_BATCHES = 9  # Number batch splits

    # Calculate the number of param combs needed
    res = 1
    for key, vals in d_params.items():
        res = res * len(vals)
    # Log relevant information on batch, combinations, and training sets
    logging.info(f"Total number of combinations generated: {res}")
    logging.info(f"Number of batches required: {math.ceil(res/N_BATCHES)}")
    logging.info(f"Count of training sets: {len(close_train_dfs)}")

    gridsearchCV(
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
