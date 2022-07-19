import logging
import numpy as np
from sklearn.model_selection import train_test_split
from optimizers.train import geneticCV
from optimizers.utils._utils import get_csv_data

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
        "period": np.arange(10, 100, 10, dtype=int),
        "upper": np.arange(2.0, 5.2, 0.1, dtype=float),
        "lower": np.arange(2.0, 5.2, 0.1, dtype=float) * -1.0,
        "exit": np.arange(0.5, 2.0, 0.1, dtype=float),
        "delta": 0.1 ** np.arange(1, 10, 1, dtype=float),
        "vt": np.arange(0.1, 1.1, 0.1, dtype=float),
    }

    opens = get_csv_data("data/hourly_open_data.csv")
    closes = get_csv_data("data/hourly_close_data.csv")

    opens, _ = train_test_split(opens, test_size=0.20, train_size=0.80, shuffle=False)
    closes, _ = train_test_split(closes, test_size=0.20, train_size=0.80, shuffle=False)
    
    df = geneticCV(
            opens, closes, params,
            n_iter=100,
            n_batch_size=10,
            population=200,
            rank_method="rank_space",
            elitism={0: 0.167, 50: 0.333, 80: 0.667},
            diversity={0: 0.667, 70: 0.333, 90: 0.000},
            cv="sliding",
            burnin=200,
            hedge="beta",
            mode="cummlog",
            max_workers=None,
            n_splits=4,
            trade_const=4,
        )

    logging.info("Genetic algorithm search completed.")
    df.to_csv("results.csv")
