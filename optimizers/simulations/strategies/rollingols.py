import numpy as np
from numba import njit
from collections import namedtuple
from vectorbt.base.reshape_fns import flex_select_auto_nb
from .components.models import OLS, discretized_OU
from .components.preprocessors import (
    log_return_transform, cummulative_return_transform, log_price_transform
)

Memory = namedtuple("Memory", ('sscore', 'signal', 'status'))       
Params = namedtuple("Params", ('period', 'upper_entry', 'upper_exit', 'lower_entry', 'lower_exit', 'order_size'))
Transformations = namedtuple("Transformations", ('logret', 'log', 'cumlog'))


@njit
def ols_pre_group_func_nb(c, _period, _upper_entry, _upper_exit, _lower_entry, _lower_exit, _order_size):
    """Preprocess data for use during backtesting"""
    assert c.group_len == 2

    sscore = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    signal = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    status = np.full(1, 0, dtype=np.int_)

    period = flex_select_auto_nb(np.asarray(_period), 0, c.group, True)
    upper_entry = flex_select_auto_nb(np.asarray(_upper_entry), 0, c.group, True)
    upper_exit = flex_select_auto_nb(np.asarray(_upper_exit), 0, c.group, True)
    lower_entry = flex_select_auto_nb(np.asarray(_lower_entry), 0, c.group, True)
    lower_exit = flex_select_auto_nb(np.asarray(_lower_exit), 0, c.group, True)
    order_size = flex_select_auto_nb(np.asarray(_order_size), 0, c.group, True)

    # Calculate the transformation upfront to reference (if needed) later
    arr = c.close[:, :c.from_col+c.group_len] # This will pull out the closes as an Nx2 array
    log = log_price_transform(arr)
    logret = log_return_transform(arr)
    cumlog = cummulative_return_transform(arr)

    memory = Memory(sscore, signal, status)
    params = Params(period, upper_entry, upper_exit, lower_entry, lower_exit, order_size)
    transformations = Transformations(logret, log, cumlog)

    size = np.empty(c.group_len, dtype=np.float_)

    return (memory, params, size, transformations)

@njit
def ols_pre_segment_func_nb(c, memory, params, size, transformations, transform, hedge, standarization):
    
    if c.i < params.period - 1:
        size[0] = np.nan
        size[1] = np.nan
        return (size,)
    
    window_slice = slice(max(0, c.i + 1 - params.period), c.i + 1)

    if not transform or transform == "default":
        x = c.close[window_slice, c.from_col]         
        y = c.close[window_slice, c.from_col + 1]
    elif transform == "cumlog":
        x = transformations.cumlog[window_slice, c.from_col]
        y = transformations.cumlog[window_slice, c.from_col + 1]
    elif transform == "logret":
        x = transformations.logret[window_slice, c.from_col]
        y = transformations.logret[window_slice, c.from_col + 1]
    elif transform == "log":
        x = transformations.log[window_slice, c.from_col]
        y = transformations.log[window_slice, c.from_col + 1]

    beta, alpha = OLS(x, y)

    resids = y - (x * beta + alpha)

    if standarization == 'sscore':
        memory.sscore[c.i] = discretized_OU(resids)
    elif standarization == 'zscore':
        memory.sscore[c.i] = (resids[-1] - resids.mean()) / resids.std()
    elif standarization == 'zscorealt':
        memory.sscore[c.i] = -resids.mean() / resids.std()
    elif standarization == 'sscorealt':
        memory.sscore[c.i] = discretized_OU(resids, alternative_calc=True)

    outlay = c.last_value[c.group] * params.order_size

    if memory.sscore[c.i - 1] > params.upper_entry and not memory.status[0]:
        if hedge == "dollar":
            size[0] = outlay / c.close[c.i - 1, c.from_col] # X asset
            size[1] = -outlay / c.close[c.i - 1, c.from_col + 1] # y asset
        elif hedge == 'beta':
            if np.abs(beta) < 1:
                size[0] = (outlay * beta) / c.close[c.i - 1, c.from_col]
                size[1] = -outlay / c.close[c.i - 1, c.from_col + 1]
            else:
                size[0] = outlay / c.close[c.i - 1, c.from_col]
                size[1] = -(outlay / beta) / c.close[c.i - 1, c.from_col + 1]

        c.call_seq_now[0] = 1 # Execute short sale first
        c.call_seq_now[1] = 0 # Use funds to purchase long side
        memory.status[0] = 1

    elif memory.sscore[c.i - 1] < params.lower_entry and not memory.status[0]:
        if hedge == "dollar":
            size[0] = -outlay / c.close[c.i - 1, c.from_col] # X asset
            size[1] = outlay / c.close[c.i - 1, c.from_col + 1] # y asset
        elif hedge == "beta":
            if np.abs(beta) < 1:
                size[0] = -(outlay * beta) / c.close[c.i - 1, c.from_col]
                size[1] = outlay / c.close[c.i - 1, c.from_col + 1]
            else:
                size[0] = -outlay / c.close[c.i - 1, c.from_col]
                size[1] = (outlay / beta) / c.close[c.i - 1, c.from_col + 1]
        
        c.call_seq_now[0] = 0
        c.call_seq_now[1] = 1
        memory.status[0] = 2

    elif memory.status[0] == 1:
        if memory.sscore[c.i - 1] < params.upper_exit:
            size[0] = 0
            size[1] = 0
            c.call_seq_now[0] = 0
            c.call_seq_now[1] = 1
            memory.status[0] = 0
        
    elif memory.status[0] == 2:
        if memory.sscore[c.i - 1] > params.lower_exit:
            size[0] = 0
            size[1] = 0
            c.call_seq_now[0] = 1
            c.call_seq_now[1] = 0
            memory.status[0] = 0

    else:
        size[0] = np.nan
        size[1] = np.nan
    
    return (size,)
