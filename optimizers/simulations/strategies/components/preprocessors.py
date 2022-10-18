import numpy as np
from numba import njit

@njit
def log_price_transform(arr:np.array) -> np.array:
    """Numba compatible logarithmic transform for NxM array"""
    log_transformations = np.full(arr.shape, 0, dtype=np.float_)

    for idx in range(0, arr.shape[1]):
        col = arr[:, idx]
        transformed_col = np.log(col)
        log_transformations[:, idx] = transformed_col

    return log_transformations

@njit 
def log_return_transform(arr:np.array) -> np.array:
    """Numba compatible logarithmic returns transformation for NxM array"""
    log_ret_transformaion = np.full(arr.shape, 0, dtype=np.float_)

    for idx in range(0, arr.shape[1]):
        col = arr[:, idx]
        logr_col = np.log1p((col[1:] - col[:-1]) / col[:-1])
        log_ret_transformaion[1:, idx] = logr_col

    return log_ret_transformaion

@njit
def cummulative_return_transform(arr:np.array) -> np.array:
    """Numba compatible cummulative logarithmic return transformation for NxM array"""
    cum_ret_transformation = np.full(arr.shape, 0, dtype=np.float_)

    for idx in range(0, arr.shape[1]):
        col = arr[:, idx]
        cumr_col = np.log1p((col[1:] - col[:-1]) / col[:-1]).cumsum()
        cum_ret_transformation[1:, idx] = cumr_col

    return cum_ret_transformation

@njit
def min_max_transform(arr:np.array) -> np.array:
    """Numba-compatible min-max function for NxM dimensional array"""
    scaled_vals = np.full(arr.shape, 0, dtype=np.float_)

    for idx in range(0, arr.shape[1]):
        col = arr[:, idx]
        scaled_col = (col - col.min()) / (col.max() - col.min())
        scaled_vals[:, idx] = scaled_col

    return scaled_vals
