import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def englegranger(x, y, trend="c", maxlag=1):
    """Perform engle-granger cointegration test on data stores in PSQL database"""
    rres = coint(x, y, maxlag=maxlag, trend=trend)
    lres = coint(y, x, maxlag=maxlag, trend=trend)
    if rres[0] <= lres[0]:
        x, y = x.name.lower(), y.name.lower()
        t = rres[0]
        p = rres[1]
    else:
        x, y = y.name.lower(), x.name.lower()
        t = lres[0]
        p = lres[1]
    res = {
        "x": x, "y": y,
        "tscore": t,
        "pvalue": p
    }
    return res

def halflife(residuals):
    """Calculate half life of mean reversion via OLS"""
    rlag = residuals.shift(1).dropna()
    deltaResids = (residuals[1:] - rlag).dropna()
    # Regress lagged residuals against residuals
    mod2 = sm.OLS(
        endog=deltaResids,
        exog=sm.add_constant(rlag)
    )
    res2 = mod2.fit()

    return -np.log(2)/res2.params[0] # <- params[0]= beta

def hurst(ts, maxlag=200):
    # Create the range of lag values
    lags = range(2, maxlag)

    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0
