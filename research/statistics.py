import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from numba import njit


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

def rolling_zscore_nb(resids, window):

    zscore = np.full(resids.shape[0], np.nan, dtype=np.float_)

    for i in range(0, resids.shape[0]):
        if i > window - 2:
            window_slice = slice(max(0, i + 1 - window), i + 1)
            s = resids[window_slice]
            zscore[i] = (resids[i] - np.mean(s)) / np.std(s)

    return zscore

@njit
def KF(X, y, R, C, theta, delta=1e-5, vt=1):
    """Kalman Filter as outlined by E. Chan in Algorithmic Trading pg. 78"""
    Wt = (delta / (1 - delta)) * np.eye(2)

    if np.isnan(R).any():
        R = np.ones((2,2))
    else:   
        R = C + Wt

    F = np.asarray([X, 1.0], dtype=np.float_).reshape((1,2))

    yhat = F.dot(theta) # Prediction
    et = y - yhat       # Calculate error term

    Qt = F.dot(R).dot(F.T) + vt
    At = R.dot(F.T) / Qt
    theta = theta + At.flatten() * et   # Update theta
    C = R - At * F.dot(R)

    return R, C, theta, et, Qt


def kalmanfilter(data:pd.DataFrame, delta:float=1e-5, vt:float=1.0, export_df:bool=True) -> tuple or pd.DataFrame:
    """Numba-acceralated Kalman filter wrapper"""
    theta = np.full(2, 0, dtype=np.float_)          # 2x1 matrix representing beta, intercept from filter
    Rt = np.full((2,2), np.nan, dtype=np.float_)    # Generates matrix of [[0,0],[0,0]]
    Ct = np.full((2,2), np.nan, dtype=np.float_)    # C matrix is correctly implemented here

    thetas = []
    C = []
    R = []
    errors = []

    for xt, yt in zip(data.iloc[:, 0], data.iloc[:, 1]):
        Rt, Ct, theta, et, _ = KF(xt, yt, Rt, Ct, theta, delta=delta, vt=vt)
        thetas.append(theta)
        C.append(Ct)
        R.append(Rt)
        errors.append(et)

    thetas = np.vstack(thetas)

    C = np.vstack(C)
    R = np.vstack(R)

    C = C.reshape((int(C.shape[0] / 2), C.shape[1], 2))
    R = R.reshape((int(R.shape[0] / 2), R.shape[1], 2))

    errors = np.vstack(errors)

    if export_df:
        res = pd.DataFrame(thetas, index=data.index)
        res.rename(columns={
            0: "slope",
            1: "intercept",
        }, inplace=True)
    else:
        res = thetas, C, errors, R # State_mean, state_covariance

    return res

def kelly_criterion(win_rate, profit_from_win, potential_loss):
    """Modified kelly criterion for non-total loss bet sizing"""
    kelly_pct = (win_rate/np.abs(potential_loss)) - (1 - win_rate)/np.abs(profit_from_win)
    return kelly_pct
