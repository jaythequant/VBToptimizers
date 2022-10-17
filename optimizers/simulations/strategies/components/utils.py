import numpy as np
from numba import njit

@njit
def calculate_residuals(x:np.array, y:np.array, slope:float, intercept:float) -> np.array:
    return y - (x * slope + intercept)
