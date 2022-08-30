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
        "period": np.arange(10, 1000, 10, dtype=int),
        "upper": np.arange(2.1, 5.1, 0.1, dtype=float),
        "lower": np.arange(2.1, 5.1, 0.1, dtype=float) * -1.0,
        "exit": np.arange(0.0, 2.1, 0.1, dtype=float),
        "delta": np.vstack([arr * (0.1 ** np.arange(1,10,1)) for arr in np.arange(1,10,1)]).flatten(),
        "vt": np.arange(0.01, 2.01, 0.01, dtype=float), # Experimenting with smaller step sizes
    }

    fil = "btczec"
    opens = get_csv_data(f"data/{fil}_hourly_opens.csv")
    closes = get_csv_data(f"data/{fil}_hourly_closes.csv")

    opens, _ = train_test_split(opens, test_size=0.20, train_size=0.80, shuffle=False)
    closes, _ = train_test_split(closes, test_size=0.20, train_size=0.80, shuffle=False)

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
            n_iter=30,
            n_batch_size=75,
            population=1500,
            rank_method="rank_space",
            elitism={0: 0.005, 25: 0.500},
            diversity={0: 1.00, 25: 0.200},
            cv="sliding",
            slippage=0.0010,
            burnin=300,
            hedge="beta",
            mode="log",
            n_splits=4,
            trade_const=0.210,   # Recommended a 0.225
            pr_const=0.135,      # Recommended at 0.135
            wr_const=1.470,      # Recommended at 1.350
            duration_cap=1440,   # Punish trade duration >=1 day
            order_size=0.10,
            freq="h",
        )

    logging.info("Genetic algorithm search completed.")
    df.to_csv("results.csv")
