import vectorbt as vbt
import pandas as pd
import concurrent.futures
from itertools import repeat
import logging

from setup.lqe_setup import * # Import a state space model setup implementing Kalman Filtering
from utils.cross_validators import vbt_cv_kfold_constructor
from utils.statistics import generate_random_sample, score_results, return_results

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


def simulate_from_order_func(
    close_data, open_data, period, upper, lower, exits, burnin=500, delta=1e-5, vt=1, 
    mode="Kalman", cash=100_000, commission=0.0008, slippage=0.0010, order_size=0.10, 
    freq="d"
):
    """Simulate pairs trading strategy with multiple signal strategy optionality"""
    return vbt.Portfolio.from_order_func(
        close_data,
        order_func_nb, 
        open_data.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=pre_group_func_nb, 
        pre_group_args=(
            period, 
            upper, 
            lower, 
            exits, 
            delta, 
            vt, 
            order_size, 
            burnin,
        ),
        pre_segment_func_nb=pre_segment_func_nb, 
        pre_segment_args=(mode,),
        fill_pos_record=False,  # a bit faster
        init_cash=cash,
        cash_sharing=True, 
        group_by=True,
        freq=freq
    )


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


if __name__ == "__main__":

    logging.info("Initializing training session . . . ")

    closes = pd.read_csv("data/crypto_close_data.csv", index_col="time")
    opens = pd.read_csv("data/crypto_open_data.csv", index_col="time")

    close_train_dfs, close_test_dfs = vbt_cv_kfold_constructor(closes, n_splits=5)
    open_train_dfs, open_test_dfs = vbt_cv_kfold_constructor(opens, n_splits=5)

    sample_set = generate_random_sample(n_iter=100)

    best_comb = None
    best_wr = None
    best_ret = None

    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        for result in executor.map(testParams, repeat(close_train_dfs), repeat(open_train_dfs), sample_set):
            sample, wr, ret = result # Unpack out result into params, win rate, and total return
            if not best_comb:
                best_comb = sample
                best_wr = wr
                best_ret = ret
                logging.info(best_comb)
                logging.info(f"Score: {wr:.4f}")
                logging.info(f"Return: {ret:.4f}")
            if wr > best_wr:
                best_comb = sample
                best_wr = wr
                best_ret = ret
                logging.info("New best wr")
                logging.info(f"Score: {wr:.4f}")
                logging.info(f"Return: {ret:.4f}")

    logging.info("Tests complete:")
    logging.info("Optimized parameters:", best_comb)
    logging.info(f"Score: {wr:.4f}")
    logging.info(f"Return: {ret:.4f}")
    logging.info("Shutting down training session")
