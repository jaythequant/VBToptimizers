import numpy as np
import logging
import pandas as pd
import numpy as np

from optimizers.geneticCV import geneticCV

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

    logging.info("Initializing genetic cross-validator . . . ")

    params = {
        "period": np.arange(30000, 60000, 2000, dtype=int),
        "upper": np.arange(2.0, 5.2, 0.2, dtype=float),
        "lower": np.arange(2.0, 5.2, 0.2, dtype=float) * -1.0,
        "exit": np.arange(0.5, 2.0, 0.1, dtype=float),
        "delta": 0.1 ** np.arange(5, 10, 1, dtype=float),
        "vt": np.arange(0.5, 1.1, 0.1, dtype=float),
    }   

    opens = pd.read_csv("optimizers/data/crypto_open_data.csv", index_col="time")[500_000:]
    closes = pd.read_csv("optimizers/data/crypto_close_data.csv", index_col="time")[500_000:]

    df = geneticCV(
            opens, closes, params, 
            n_iter=200, 
            n_batch_size=13, 
            population=100,
            max_workers=4,
        )

    print(df)