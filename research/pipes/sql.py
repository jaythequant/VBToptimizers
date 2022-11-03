import pandas as pd
import warnings
from sqlalchemy import create_engine
from kucoincli.client import Client


class SQLPipe:
    """Pipeline from PSQL database into pandas DataFrames"""

    def __init__(self, schema, database, username, password, interval):
        """Set initial pipeline parameters by defining a target schema and generating an engine into the SQL database"""
        self.engine = create_engine(f"postgresql+psycopg2://{username}:{password}@localhost/{database}")
        self.schema = schema
        self.interval = interval
        self.client = Client()

    @staticmethod
    def __prep_assets(assets):
        """Format asset(s) to SQL table format"""
        is_string = True if isinstance(assets, str) else False
        assets = [assets] if not isinstance(assets, (list, tuple)) else assets
        assets = [asset.replace("-", "").lower() for asset in assets]
        assets = assets[0] if is_string else assets
        return assets

    def query_asset(self, asset, only_close=False, ascending=True, warn=True):
        """Query asset as pandas DataFrame from SQL database"""
        pair = self.__prep_assets(asset)
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
        tables = self.__prep_assets(assets)
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
        tables = self.__prep_assets(assets)
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
        self, stablepairs:bool=True, leveragetokens:bool=True, only_marginable:bool=False,
        row_min:int=0, quote_curr=None,
    ) -> list:
        """Query the database for list of all asset tables

        Parameters
        ----------
        stablepairs : bool, optional
            Set `stablepairs=False` to remove stablevalue-to-stable value trading pairs (e.g. USDC-USDT).
        leveragetokens : bool, optional
            Set `leveragetokens=False` to remove 3x leverages tokens (e.g. ETH3S-USDT).
        only_marginable : bool, optional
            Set `only_marginable=True` to include only marginable trading pairs.
        row_min : int, optional
            Filter list by minimum number of rows in table.
        quote_curr : str or list, optional
            Filter return by specified quote currency or currencies

        Returns
        -------
        list
            Filtered list of all table names for trading pairs fitting specified parameters
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
        if only_marginable:
            marginable_symbols = self.client.symbols(marginable=True).index.str.replace("-", "").str.lower()
            symbols = [symbol for symbol in symbols if symbol in marginable_symbols]
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
            df = self.client.all_tickers()
            conditions = (df['high'].astype(float) < 1.01) & (df['low'].astype(float) > 0.990)
            stablepairs = df[conditions].index.str.replace("-", "").str.lower()
            symbols = [symbol for symbol in symbols if symbol not in stablepairs]
        if quote_curr:
            df = self.client.symbols(quote=quote_curr)
            valid_assets = self.__prep_assets(df.index.to_list())
            symbols = [symbol for symbol in symbols if symbol in valid_assets]
        return symbols
