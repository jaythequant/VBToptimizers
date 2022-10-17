import numpy as np
from numba import njit
from collections import namedtuple
from vectorbt.base.reshape_fns import flex_select_auto_nb
from .components.predictors import OLS
from .components.utils import calculate_residuals

Memory = namedtuple("Memory", ('sscore', 'signal', 'status'))       
Params = namedtuple("Params", ('period', 'upper_entry', 'upper_exit', 'lower_entry', 'lower_exit', 'order_size'))
Transformations = namedtuple("Transformations", ("xret", "yret"))


@njit
def estimate_ou_process(residuals:np.array) -> float:
    """Estimate Ornstein-Uhlenbeck process from OLS residuals; return s-score"""

    Xk = residuals.cumsum()
    b, a = OLS(Xk[:-1], Xk[1:])
    resids = Xk[1:] - (Xk[:-1] * b + a)

    kappa = -np.log(b) * (365*24)
    m = a / (1-b)
    sigma_eq = np.sqrt(np.var(resids)/(1-b**2))

    return m / sigma_eq # s-score

@njit
def ou_pre_group_func_nb(c, _period, _upper_entry, _upper_exit, _lower_entry, _lower_exit, _order_size):
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

    logr_x = np.full(c.close[:, c.from_col].shape, 0, dtype=np.float_)
    logr_y = np.full(c.close[:, c.from_col +1].shape, 0, dtype=np.float_)
    logr_x[1:] = np.log1p((c.close[:, c.from_col][1:] - c.close[:, c.from_col][:-1]) / c.close[:, c.from_col][:-1])
    logr_y[1:] = np.log1p((c.close[:, c.from_col +1][1:] - c.close[:, c.from_col +1][:-1]) / c.close[:, c.from_col +1][:-1])

    memory = Memory(sscore, signal, status)
    params = Params(period, upper_entry, upper_exit, lower_entry, lower_exit, order_size)
    transformations = Transformations(logr_x, logr_y)

    size = np.empty(c.group_len, dtype=np.float_)

    return (memory, params, size, transformations)

@njit
def ou_pre_segment_func_nb(c, memory, params, size, transformations):
    
    if c.i < params.period - 1:
        size[0] = np.nan
        size[1] = np.nan
        return (size,)
    
    window_slice = slice(max(0, c.i + 1 - params.period), c.i + 1)

    x = transformations.xret[window_slice]
    y = transformations.yret[window_slice]
    beta, alpha = OLS(x, y)
    
    resids = y - (x * beta + alpha)
    memory.sscore[c.i] = estimate_ou_process(resids)

    outlay = c.last_value[c.group] * params.order_size

    if memory.sscore[c.i - 1] > params.upper_entry and not memory.status[0]:
        size[0] = (outlay * beta) / c.close[c.i - 1, c.from_col]
        size[1] = -outlay / c.close[c.i - 1, c.from_col + 1]

        c.call_seq_now[0] = 1 # Execute short sale first
        c.call_seq_now[1] = 0 # Use funds to purchase long side
        memory.status[0] = 1

    elif memory.sscore[c.i - 1] < params.lower_entry and not memory.status[0]:
        size[0] = -(outlay * beta) / c.close[c.i - 1, c.from_col]
        size[1] = outlay / c.close[c.i - 1, c.from_col + 1]
        
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
