from matplotlib.transforms import Transform
import numpy as np
from numba import njit
from collections import namedtuple
from vectorbt.base.reshape_fns import flex_select_auto_nb
from .components.preprocessors import min_max_transform
from .components.preprocessors import (
    log_price_transform, log_return_transform, cummulative_return_transform,
)

Memory = namedtuple("Memory", ('spread', 'signal', 'status'))       
Params = namedtuple("Params", ('period', 'upper', 'lower', 'long_exit', 'short_exit', 'order_size'))
Transformations = namedtuple("Transformations", ("log", "logret", "cumlog"))


@njit
def ssd_pre_group_func_nb(c, _period, _long_entry, _long_exit, _short_entry, _short_exit, _order_size):
    """Preprocess data for use during backtesting"""
    assert c.group_len == 2

    spread = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    signal = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    status = np.full(1, 0, dtype=np.int_)

    # Calculate the transformation upfront to reference (if needed) later
    arr = c.close[:, :c.from_col+c.group_len] # This will pull out the closes as an Nx2 array
    log = log_price_transform(arr)
    logret = log_return_transform(arr)
    cumlog = cummulative_return_transform(arr)

    transformations = Transformations(log, logret, cumlog) # Store the transformations here

    period = flex_select_auto_nb(np.asarray(_period), 0, c.group, True)
    long_entry = flex_select_auto_nb(np.asarray(_long_entry), 0, c.group, True)
    long_exit = flex_select_auto_nb(np.asarray(_long_exit), 0, c.group, True)
    short_entry = flex_select_auto_nb(np.asarray(_short_entry), 0, c.group, True)
    short_exit = flex_select_auto_nb(np.asarray(_short_exit), 0, c.group, True)
    order_size = flex_select_auto_nb(np.asarray(_order_size), 0, c.group, True)

    memory = Memory(spread, signal, status)
    params = Params(period, long_entry, long_exit, short_entry, short_exit, order_size)

    size = np.empty(c.group_len, dtype=np.float_)

    return (memory, params, size, transformations)

@njit
def ssd_pre_segment_func_nb(c, memory, params, size, transformations, mode, hedge):
    
    if c.i < params.period - 1:
        size[0] = np.nan
        size[1] = np.nan
        return (size,)
    
    window_slice = slice(max(0, c.i + 1 - params.period), c.i + 1)

    if mode == "default":
        X = c.close[window_slice, c.from_col]         
        y = c.close[window_slice, c.from_col + 1]
    elif mode == "cumlog":
        X = transformations.cumlog[window_slice, c.from_col] 
        y = transformations.cumlog[window_slice, c.from_col + 1]
    elif mode == "logret":
        X = transformations.logret[window_slice, c.from_col] 
        y = transformations.logret[window_slice, c.from_col + 1]
    elif mode == "log":
        X = transformations.log[window_slice, c.from_col] 
        y = transformations.log[window_slice, c.from_col + 1]
