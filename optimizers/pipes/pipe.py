import pandas as pd
import warnings
from sqlalchemy import create_engine
from ._exceptions import HistoricDataError


class SQLPipe:
    """Pipeline from PSQL database into pandas DataFrames"""

    def __init__(self, schema, database, username, password, interval='60T'):

        self.engine = create_engine(f"postgresql+psycopg2://{username}:{password}@localhost/{database}")
        self.schema = schema
        self.interval = interval

    def query_pair(self, pair, only_close=False, ascending=True, raise_warning=True):
        if only_close:
            get_pair = f"""
                SELECT
                {self.db}."{pair}".close
                FROM {self.db}."{pair}"
                """
        get_pair = f"""
            SELECT *
            FROM {self.schema}."{pair}"
            """
        df = pd.read_sql(get_pair, self.engine, index_col="time")
        df.index = pd.to_datetime(df.index)
        # Quick check for missing timeseries data + warning message
        missing_data = pd.date_range(
                start=df.index.min(), end=df.index.max(), freq=self.freq
            ).difference(df.index)
        if raise_warning:
            if len(missing_data) > 0:
                warnings.warn(
                    f"""\nData Warning: Timeseries index contains missing intervals
                    Missing {len(missing_data)} time intervals
                    """)
        return df.sort_index(ascending=ascending)

    def query_pairs_trading_backtest(self, assets:list):
        """Query set of assets for use with pairs trading backtesting framework"""
        assets = [assets] if isinstance(assets, str) else assets
        tables = [asset.replace("-", "").lower() for asset in assets]

        dfs = []

        for table in tables:
            query = f"""
                SELECT time, close, open
                FROM "{self.schema}"."{table}"
            """
            df = pd.read_sql(query, self.engine, index_col="time")
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)
            dfs.append(df)

        df = pd.concat(dfs, axis=1, keys=assets).dropna()

        missing_data = pd.date_range(
                start=df.index.min(), end=df.index.max(), freq=self.interval
            ).difference(df.index)
        if len(missing_data) > 0:
            raise HistoricDataError(f'Missing {len(missing_data)} bars in dataset')
            
        return df
