import logging
import pandas as pd
from optimizers.train import randomSearch

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


if __name__ == "__main__":

    logging.info("Initializing training session . . . ")

    closes = pd.read_csv("optimizers/data/crypto_close_data.csv", index_col="time")[:700_000]
    opens = pd.read_csv("optimizers/data/crypto_open_data.csv", index_col="time")[:700_000]

    best_comb, wr, ret = randomSearch(closes, opens, workers=4, n_iter=5000, n_splits=5)

    logging.info("Tests complete:")
    logging.info("Optimized parameters:", best_comb)
    logging.info(f"Score: {wr:.4f}")
    logging.info(f"Return: {ret}")
    logging.info("Shutting down training session")