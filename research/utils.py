import pandas as pd
import datetime as dt
from kucoincli.client import Client


def test_lending_liquidity(quote='USDT'):
    """Obtain max point-in-time liquidity for lending markets in USDT terms"""
    client = Client()
    l = client.symbols(marginable=True).baseCurrency
    liq = {}

    for curr in l:
        try:
            df = client.lending_rate(curr)
            stats = client.get_stats(curr + '-' + quote)
            max_borrow = (((stats.buy + stats.sell) / 2) * df['size'].sum())
            liq[curr] = max_borrow
        except:
            pass

    return pd.Series(liq).sort_values(ascending=False)

def test_trading_liquidity(lookback=90, interval='1day'):
    """Calculate mean turnover for marginable currencies in `interval` granularity over `lookback` days"""
    client = Client()
    l = client.symbols(marginable=True).index
    liq = {}

    start = dt.datetime.now() - dt.timedelta(days=lookback)
    
    for curr in l:
        mean_vol = client.ohlcv(
            tickers=curr, 
            interval=interval,
            start=start
        ).turnover.mean()
        liq[curr] = mean_vol
    
    return pd.Series(liq).sort_values(ascending=False)
