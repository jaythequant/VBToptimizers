import logging
import os
import pandas as pd
import numpy as np
import concurrent.futures
import statsmodels.api as sm
from itertools import combinations, repeat
from statistics import (englegranger, hurst, halflife)
from sqlalchemy import create_engine
from dotenv import load_dotenv
from kucoincli.client import Client
from pipes.sql import SQLPipe

load_dotenv()

USER = os.getenv('psql_username')
PASS = os.getenv('psql_password')
DATABASE = 'crypto'
SCHEMA = 'bihourly'
INTERVAL = '30T'

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_marginable(client):
    """Get list of marginable securites"""
    tickers = client.symbols(marginable=True).index.to_list()
    tickers = [ticker.replace("-", "").lower() for ticker in tickers]
    return tickers


def get_historic_data(assets, schema, engine, interval):
    """Query SQL database for historic data"""
    assets = [assets] if isinstance(assets, str) else assets
    tables = [asset.replace("-", "").lower() for asset in assets]

    dfs = []

    for table in tables:
        query = f"""
            SELECT time, close, open
            FROM "{schema}"."{table}"
        """
        df = pd.read_sql(query, engine, index_col="time")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)
        idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=interval)
        df = df.reindex(idx).interpolate(limit=1)
        dfs.append(df)

    df = pd.concat(dfs, axis=1, keys=assets)
    # Locate final NaN value in timeseries and slice out all prior data
    df = df.loc[df[df.isna().any(axis=1)].index.max():, :][1:]
    df = df.xs('close', level=1, axis=1)

    return df


def test_pairs(comb, schema, interval, min_rows=8640, slice=1000, transformation="log"):
    """Apply series of statistical tests to two asset combinations"""
    engine = create_engine(f"postgresql+psycopg2://james:password@localhost/crypto")
    df = get_historic_data(comb, schema, engine, interval).iloc[slice:, :]
    if df.shape[0] < min_rows:
        return
    if transformation == "log":
        df = np.log(df).dropna()
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    results = englegranger(x, y)
    mod = sm.OLS(
        endog=df.loc[:,results["y"]],
        exog=sm.add_constant(df.loc[:,results["x"]])
    )
    res = mod.fit()
    results["hurst"] = hurst(res.resid.values)
    results["halflife"] = halflife(res.resid)
    return results


if __name__ == "__main__":

    # All of these should be argparse flags?
    transformation = "log"
    trend = "c"
    maxlag = 1
    drop_n_rows = 1000
    min_rows = 17280

    pipe = SQLPipe(SCHEMA, DATABASE, USER, PASS, INTERVAL)

    logging.info("Initializing tests")
    engine = create_engine("postgresql+psycopg2://james:password@localhost/crypto")

    client = Client()
    l = pipe.get_symbol_list(
        stablepairs=False,
        leveragetokens=False,
        price_currency="usdt",
        row_min=17280,
    )
    
    marginable = get_marginable(client)
    l = list(set(l) & set(marginable))
    combs = list(combinations(l, 2))
    logging.info(f"Produced {len(combs)} unique combinations")
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        for result in executor.map(
            test_pairs, combs, 
            repeat(SCHEMA), 
            repeat(INTERVAL), 
            repeat(min_rows), 
            repeat(drop_n_rows),
            repeat(transformation)
        ):
            if result:
                results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("results.csv")
    logging.info("Statistical testing completed")
