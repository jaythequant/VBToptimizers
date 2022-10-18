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
    # np.linalg.inv raises warning. Need to fix this in the future
    _X = np.vstack((X, np.ones(len(X)))).T
    params = np.dot(np.linalg.inv(np.dot(_X.T, _X)), np.dot(_X.T, y))
    return params


@njit
def discretized_OU(residuals:np.array, alternative_calc:bool=False) -> float:
    """Estimate Ornstein-Uhlenbeck process from OLS residuals; return s-score"""

    Xk = residuals.cumsum()
    params = OLS(Xk[:-1], Xk[1:]) # Returns [beta, alpha]
    resids = Xk[1:] - (Xk[:-1] * params[0] + params[1])

    # kappa = -np.log(b) * (365*24) <- Kappa is unused for our purposes
    m = params[1] / (1-params[0])
    sigma_eq = np.sqrt(np.var(resids)/(1-params[0]**2))
    if alternative_calc:
        # This is purely experimental, may not be statistically valid
        s = (residuals[-1] - m) / sigma_eq
    else:
        s = -m / sigma_eq

    return s
