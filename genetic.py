import logging
import configparser
import numpy as np
from sklearn.model_selection import train_test_split
from optimizers.train import geneticCV
from optimizers.utils._utils import get_csv_data

config = configparser.ConfigParser()
config.read("conf.ini")
genetic = dict(config["genetic"])
model = dict(config["backtest"])
compute = dict(config["compute"])
validation = dict(config["validation"])

# Logging config
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
FMT = "%(asctime)s [%(levelname)s] %(module)s :: %(message)s"
logging.basicConfig(
    format=FMT,
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(filename="train.log"),
        stream_handler,
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logging.info("Initializing genetic cross-validator . . . ")

    params = {
        "period": np.arange(10, 500, 10, dtype=int),
        "upper": np.arange(2.1, 4.1, 0.1, dtype=float),
        "lower": np.arange(2.1, 4.1, 0.1, dtype=float) * -1.0,
        "exit": np.arange(0.0, 2.0, 0.1, dtype=float),
        "delta": 0.1 ** np.arange(1, 10, 1, dtype=float),
        "vt": np.arange(0.1, 1.6, 0.1, dtype=float),
    }

    opens = get_csv_data("data/fileos_hourly_opens.csv")
    closes = get_csv_data("data/fileos_hourly_closes.csv")

    opens, _ = train_test_split(opens, test_size=0.30, train_size=0.70, shuffle=False)
    closes, _ = train_test_split(closes, test_size=0.30, train_size=0.70, shuffle=False)

    logging.info(f"""
    +-- Genetic Algorithm --+ +-- Model Selection --+ +-- Compute Handling -----+
    |  * Population= 1000   | |  * Population= 1000 | |  * Max Proccesses= 1000 |
    |  * Iterations= 50     | |  * Population= 1000 | |  * Population= 1000     |
    |  * Mode= cummlog      | |  * Population= 1000 | |  * Population= 1000     |
    |  * Hedge= beta        | |  * Population= 1000 | |  * Population= 1000     |
    |  * CV= sliding        | |  * Population= 1000 | |  * Population= 1000     |
    +-----------------------+ +---------------------+ +-------------------------+
    """)

    df = geneticCV(
            opens, closes, params,
            n_iter=50,
            n_batch_size=75,
            population=1500,
            rank_method="rank_space",
            elitism={0: 0.010, 42: 0.500},
            diversity={0: 1.00, 42: 0.200},
            cv="sliding",
            slippage=0.0005,
            burnin=300,
            hedge="beta",
            mode="cummlog",
            n_splits=3,
            trade_const=0.26,
            pr_const=0.30,
            wr_const=1.5,
            ret_const=1.0,
            duration_cap=1500,
            order_size=0.10,
        )

    logging.info("Genetic algorithm search completed.")
    df.to_csv("results.csv")
