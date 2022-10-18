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
stream_handler.setLevel(logging.DEBUG)
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

    arr = np.vstack([arr * (0.1 ** np.arange(1,21,1)) for arr in np.arange(1,21,1)]).flatten()
    delta_arr = np.unique(arr[arr < 1])
    vt_arr = np.unique(arr[arr < 1])

    params = {
        "entry": np.arange(0.5, 5.1, 0.05, dtype=float),
        "exit": np.arange(0.0, 3.1, 0.05, dtype=float),
        "delta": delta_arr, # large array of extremely small values
        "vt": vt_arr,
    }

    fil = "frontgrt"
    opens = get_csv_data(f"data/{fil}_hourly_opens.csv")
    closes = get_csv_data(f"data/{fil}_hourly_closes.csv")

    test_size = 0.30
    opens, _ = train_test_split(opens, test_size=test_size, train_size=(1-test_size), shuffle=False)
    closes, _ = train_test_split(closes, test_size=test_size, train_size=(1-test_size), shuffle=False)

    # logging.info(f"""
    # +-- Genetic Algorithm --+ +-- Model Selection --+ +-- Compute Handling -----+
    # |  * Population= 1000   | |  * Population= 1000 | |  * Max Proccesses= 1000 |
    # |  * Iterations= 50     | |  * Population= 1000 | |  * Population= 1000     |
    # |  * Mode= cummlog      | |  * Population= 1000 | |  * Population= 1000     |
    # |  * Hedge= beta        | |  * Population= 1000 | |  * Population= 1000     |
    # |  * CV= sliding        | |  * Population= 1000 | |  * Population= 1000     |
    # +-----------------------+ +---------------------+ +-------------------------+
    # """)

    df = geneticCV(
            opens, closes, params,
            n_iter=30,
            n_batch_size=75,
            population=1500,
            rank_method="rank_space",
            elitism={0: 0.005, 25: 0.500},
            diversity={0: 2.00, 25: 0.200},
            cv="sliding",
            slippage=0.0010,
            hedge="dollar",
            transformation=None,
            n_splits=3,
            trade_const=0.250,   # Recommended a 0.200
            sr_const=1.800,      # Recommended at 1.600
            wr_const=0.100,      # Recommended at 0.050
            trade_floor=10,
            freq='h',
            model='LQE2',
            burnin=800,
            order_size=0.10,
        )

    logging.info("Genetic algorithm search completed.")
    df.to_csv("results.csv")
