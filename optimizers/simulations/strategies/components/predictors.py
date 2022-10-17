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


@njit
def OLS(X:np.array, y:np.array) -> tuple:
    """Vectorized OLS regression (fully numba compatible)"""
    _X = np.vstack((X, np.ones(len(X)))).T
    slope, intercept = np.dot(np.linalg.inv(np.dot(_X.T, _X)), np.dot(_X.T, y))
    return slope, intercept
