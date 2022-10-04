import logging
import os
import configparser
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from optimizers.train import geneticCV
from optimizers.pipes.pipe import SQLPipe

load_dotenv()

USER = os.getenv('psql_username')
PASS = os.getenv('psql_password')
DATABASE = 'crypto'
SCHEMA = 'kucoin'

config = configparser.ConfigParser()
config.read("geneticconf.ini")
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

pipe = SQLPipe(SCHEMA, DATABASE, USER, PASS)

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logging.info("Initializing genetic cross-validator . . . ")

    params = {
        "period": np.arange(5, 1005, 5, dtype=int),
        "upper": np.arange(0.5, 5.1, 0.1, dtype=float),
        "lower": np.arange(0.5, 5.1, 0.1, dtype=float) * -1.0,
        "exit": np.arange(0.0, 3.1, 0.1, dtype=float),
        "delta": np.unique(np.vstack([arr * (0.1 ** np.arange(1,10,1)) for arr in np.arange(1,10,1)]).flatten()),
        "vt": np.unique(np.vstack([arr * (0.1 ** np.arange(1,11,1)) for arr in np.arange(1,21,1)]).flatten()),
    }

    assets = ['BNB-USDT', 'API3-USDT']
    slicer = 1000 # Slice off first few months of trading to reduce early volatility

    df = pipe.query_pairs_trading_backtest(assets)
    closes = df.xs('close', level=1, axis=1)[slicer:]
    opens = df.xs('open', level=1, axis=1)[slicer:]
    assert closes.shape[0] > 8640, 'Less than 1 year of backtesting data present'
    assert closes.shape == opens.shape, 'Open and close shape does not match'

    opens, _ = train_test_split(opens, test_size=float(validation['testsize']), train_size=float(validation['trainsize']), shuffle=False)
    closes, _ = train_test_split(closes, test_size=float(validation['testsize']), train_size=float(validation['trainsize']), shuffle=False)

    df = geneticCV(
            opens, closes, params,
            n_iter=30,
            n_batch_size=75,
            population=1500,
            rank_method="rank_space",
            elitism={0: 0.005, 25: 0.500},
            diversity={0: 2.00, 25: 0.200},
            cv="sliding",
            slippage=0.0020,
            burnin=500,
            mode="log",
            hedge="beta",
            n_splits=3,
            trade_const=0.235,   # Recommended a 0.225
            sr_const=1.230,      # Recommended at 0.350
            wr_const=1.000,      # Recommended at 1.350
            trade_floor=30,
            model='LQE1',
            freq="h"
        )

    logging.info("Genetic algorithm search completed.")
    df.to_csv("results.csv")
