import pandas as pd
import warnings
import itertools
from sqlalchemy import create_engine


class SQLPipe:
    """Pipeline from PSQL database into pandas DataFrames"""

    def __init__(self, schema, database, username, password, interval):

        self.engine = create_engine(f"postgresql+psycopg2://{username}:{password}@localhost/{database}")
        self.schema = schema
        self.interval = interval

    def query_pair(self, pair, only_close=False, ascending=True, warn=True):
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
        idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=self.interval)
        df = df.reindex(idx)
        if warn:
            if df[df.isna().any(axis=1)].shape[0]:
                warnings.warn(
                    "\nData Warning: Timeseries index missing " +
                    f"{df[df.isna().any(axis=1)].shape[0]} bars of historic data")
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
            idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=self.interval)
            df = df.reindex(idx).interpolate(limit=1)
            dfs.append(df)

        df = pd.concat(dfs, axis=1, keys=assets)
        # Locate final NaN value in timeseries and slice out all prior data
        df = df.loc[df[df.isna().any(axis=1)].index.max():, :][1:]

        return df

    def check_missing_data(self, assets:list):
        """Check for missing bars of data in asset list historic time series"""

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

        return pd.Series(missing_data)

    def query(self, query:str):
        """Query SQL database"""
        return pd.read_sql(query, self.engine)

    def get_symbol_list(
        self, stablepairs=True, leveragetokens=True, price_currency=None, 
        fiatpair=True, row_min=0, volume=0,
    ):
        """
        Query the database for all trading pairs
        :self.db: This param is the schema in my SQL db.
        :self.engine: Engine is SQLAlchemy self.engine object.
        :row_min: Filters number of rows in table. Note: 45,000 minutes in 1 month.
        """
        query_table_lengths = f"""
                SELECT 
                    nspname AS schemaname,relname,reltuples
                FROM pg_class C
                LEFT JOIN pg_namespace N ON (N.oid = C.relnamespace)
                WHERE 
                    nspname NOT IN ('pg_catalog', 'information_schema') AND
                    relkind='r' AND
                    nspname = '{self.schema}'
                ORDER BY reltuples DESC;
                """
        table_lengths = pd.read_sql(query_table_lengths, self.engine)
        symbols = list(table_lengths[table_lengths["reltuples"] >= row_min]["relname"])
        if leveragetokens == False:
            short_lever = "3s"
            long_lever = "3l"
            symbols = [
                symbol
                for symbol in symbols
                if long_lever not in symbol
                if short_lever not in symbol
            ]
        if stablepairs == False:
            stabletokens = [
                "USDT",
                "USDC",
                "TUSD",
                "EUSD",
                "GUSD",
                "BUSD",
                "PAX",
                "SUSD",
                "DAI",
                "USDN",
                "UST",
                "CUSD",
                "OUSD",
                "USDJ",
            ]
            stablepairs = list(itertools.permutations(stabletokens, 2))
            stablepairs = ["".join(tup).lower() for tup in stablepairs]
            symbols = [symbol for symbol in symbols if symbol not in stablepairs]
        if price_currency != None:
            symbols = [symbol for symbol in symbols if price_currency.lower() in symbol]
        if volume:
            stats = self.all_tickers().volValue.astype(float)
            stats.index = stats.index.str.replace("-", "").str.lower()
            symbols = [s for s in symbols if s in stats[stats >= volume]]            
        return symbols

