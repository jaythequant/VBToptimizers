import logging
import os
import configparser
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from optimizers.ga import geneticCV
from research.pipes.sql import SQLPipe

load_dotenv()

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


def main():

    USER = os.getenv('PSQL_USERNAME')
    PASS = os.getenv('PSQL_PASSWORD')
    DATABASE = 'crypto'
    SCHEMA = 'hourly'
    INTERVAL = '60T'
    ASSETS = ['SUN-USDT', 'OCEAN-USDT']
    SLICER = -7420 # This is approx. 18-months of 30 minute granularity data

    pipe = SQLPipe(SCHEMA, DATABASE, USER, PASS, INTERVAL)

    logging.info("Initializing genetic cross-validator . . . ")

    params = {
        "period": np.arange(20, 2005, 5, dtype=int),
        "upper": np.arange(0.5, 3.6, 0.1, dtype=float),
        "lower": np.arange(0.5, 3.6, 0.1, dtype=float) * -1.0,
        "exit": np.arange(0.0, 2.1, 0.1, dtype=float),
        "delta": np.unique(np.vstack([arr * (0.1 ** np.arange(1,10,1)) for arr in np.arange(1,10,1)]).flatten()),
        "vt": np.unique(np.vstack([arr * (0.1 ** np.arange(1,11,1)) for arr in np.arange(1,21,1)]).flatten()),
    }

    df = pipe.query_pairs_trading_backtest(ASSETS)
    closes = df.xs('close', level=1, axis=1)[SLICER:]
    opens = df.xs('open', level=1, axis=1)[SLICER:]
    assert closes.shape[0] > 1000, 'Less minimium required backtesting data present'
    assert closes.index.equals(opens.index), 'Open and close indices do not match'

    opens, _ = train_test_split(opens, test_size=float(validation['testsize']), train_size=float(validation['trainsize']), shuffle=False)
    closes, _ = train_test_split(closes, test_size=float(validation['testsize']), train_size=float(validation['trainsize']), shuffle=False)

    df = geneticCV(
            opens, closes, params,
            n_iter=20,
            n_batch_size=50,
            population=1000,
            rank_method="rank_space",
            elitism={0: 0.005, 15: 0.500},
            diversity={0: 2.00, 15: 0.200},
            cv="sliding",
            slippage=0.0010,
            burnin=200,
            transformation="log",
            hedge="beta",
            n_splits=3,
            trade_const=0.240,   # Recommended a 0.235
            sr_const=1.155,      # Recommended at 1.100
            wr_const=0.300,      # Recommended at 0.350
            trade_floor=40,
            model='LQE',
            freq=INTERVAL,
            standard_score='zscore',
            seed_filter=True,
        )

    logging.info("Genetic algorithm search completed.")


if __name__ == "__main__":
    main()
