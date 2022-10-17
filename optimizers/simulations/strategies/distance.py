import numpy as np
from numba import njit
from collections import namedtuple
from vectorbt.base.reshape_fns import flex_select_auto_nb

Memory = namedtuple("Memory", ('spread', 'signal', 'status'))       
Params = namedtuple("Params", ('period', 'upper', 'lower', 'long_exit', 'short_exit', 'order_size'))


@njit
def ssd_pre_group_func_nb(c, _period, _long_entry, _long_exit, _short_entry, _short_exit, _order_size):
    """Preprocess data for use during backtesting"""
    assert c.group_len == 2

    spread = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    signal = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    status = np.full(1, 0, dtype=np.int_)

    period = flex_select_auto_nb(np.asarray(_period), 0, c.group, True)
    long_entry = flex_select_auto_nb(np.asarray(_long_entry), 0, c.group, True)
    long_exit = flex_select_auto_nb(np.asarray(_long_exit), 0, c.group, True)
    short_entry = flex_select_auto_nb(np.asarray(_short_entry), 0, c.group, True)
    short_exit = flex_select_auto_nb(np.asarray(_short_exit), 0, c.group, True)
    order_size = flex_select_auto_nb(np.asarray(_order_size), 0, c.group, True)

    memory = Memory(spread, signal, status)
    params = Params(period, long_entry, long_exit, short_entry, short_exit, order_size)

    size = np.empty(c.group_len, dtype=np.float_)

    return (memory, params, size)

@njit
def ssd_pre_segment_func_nb(c, memory, params, size, transformations, mode, hedge):
    pass
