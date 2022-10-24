import pandas as pd
import numpy as np
from numba import njit


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
    errors = []

    for xt, yt in zip(data.iloc[:, 0], data.iloc[:, 1]):
        Rt, Ct, theta, et, _ = KF(xt, yt, Rt, Ct, theta, delta=delta, vt=vt)
        thetas.append(theta)
        C.append(Ct)
        errors.append(et)

    thetas = np.vstack(thetas)
    C = np.vstack(C)
    C = C.reshape((int(C.shape[0] / 2), C.shape[1], 2))
    errors = np.vstack(errors)

    if export_df:
        res = pd.DataFrame(thetas, index=data.index)
        res.rename(columns={
            0: "slope",
            1: "intercept",
        }, inplace=True)
    else:
        res = thetas, C, errors # State_mean, state_covariance

    return res
