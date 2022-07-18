import numpy as np
import utils.regs
from numba import njit
from collections import namedtuple

from vectorbt.portfolio import nb as portfolio_nb
from vectorbt.base.reshape_fns import flex_select_auto_nb
from vectorbt.portfolio.enums import SizeType


Memory = namedtuple("Memory", ('spread', 'zscore', 'beta', 'intercept', 'status'))
Params = namedtuple("Params", ('period', 'upper', 'lower', 'exit', 'order_size'))

@njit
def pre_group_func_nb(c, _period, _upper, _lower, _exit, _ordersize):
    """Prepare the current group (= pair of columns)."""

    # Context is GroupContext

    assert c.group_len == 2
    
    # In contrast to bt, we don't have a class instance that we could use to store arrays,
    # so let's create a namedtuple acting as a container for our arrays
    # ( you could also pass each array as a standalone object, but a single object is more convenient)
    spread = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    zscore = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    beta = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    intercept = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    
    # Note that namedtuples aren't mutable, you can't simply assign a value,
    # thus make status variable an array of one element for an easy assignment
    status = np.full(1, 0, dtype=np.int_)

    memory = Memory(spread, zscore, beta, intercept, status)
    
    # Treat each param as an array with value per group, and select the combination of params for this group
    period = flex_select_auto_nb(np.asarray(_period), 0, c.group, True)
    upper = flex_select_auto_nb(np.asarray(_upper), 0, c.group, True)
    lower = flex_select_auto_nb(np.asarray(_lower), 0, c.group, True)
    exit = flex_select_auto_nb(np.asarray(_exit), 0, c.group, True)
    order_size = flex_select_auto_nb(np.asarray(_ordersize), 0, c.group, True)
    
    # Put all params into a container (again, this is optional)
    params = Params(period, upper, lower, exit, order_size)
    
    # Create an array that will store our two target percentages used by order_func_nb
    # we do it here instead of in pre_segment_func_nb to initialize the array once, instead of in each row
    size = np.empty(c.group_len, dtype=np.float_)

    # The returned tuple is passed as arguments to the function below
    return (memory, params, size)
    

@njit
def pre_segment_func_nb(c, memory, params, size, mode, use_log):
    """Prepare the current segment (= row within group)."""
    
    # We want to perform calculations once we reach full window size
    if c.i < params.period - 1:
        size[0] = np.nan  # size of nan means no order
        size[1] = np.nan
        return (size,)
    
    # z-score is calculated using a window (=period) of spread values
    # This window can be specified as a slice
    window_slice = slice(max(0, c.i + 1 - params.period), c.i + 1)
    
    # Here comes the same as in rolling_ols_zscore_nb
    if mode == 'OLS':
        a = c.close[window_slice, c.from_col]
        b = c.close[window_slice, c.from_col + 1]
        memory.spread[c.i], memory.beta[c.i], memory.intercept[c.i] = utils.regs._ols_nb(a, b, use_log=use_log)
    elif mode == 'log_return':
        logret_a = np.log(c.close[c.i, c.from_col] / c.close[c.i - 1, c.from_col])
        logret_b = np.log(c.close[c.i, c.from_col + 1] / c.close[c.i - 1, c.from_col + 1])
        memory.spread[c.i] = logret_a - logret_b
    else:
        raise ValueError("Unknown mode")

    spread_mean = np.mean(memory.spread[window_slice])
    spread_std = np.std(memory.spread[window_slice])
    memory.zscore[c.i] = (memory.spread[c.i] - spread_mean) / spread_std
    
    # Used to calculate appropriate hedge ratio when intercept not forced to zero
    # hedge ratio calculated as y-hat / x_i
    if use_log:
        yhat = np.exp(np.log(c.close[c.i - 1, c.from_col]) * memory.beta[c.i - 1] + memory.intercept[c.i - 1])
    else:
        yhat = c.close[c.i - 1, c.from_col] * memory.beta[c.i - 1] + memory.intercept[c.i - 1]

    outlay = (c.last_value * params.order_size) # Determine cash outlay based on pct order_size input

    # Check if any bound is crossed
    # Since zscore is calculated using close, use zscore of the previous step
    # This way we are executing signals defined at the previous bar
    if memory.zscore[c.i - 1] > params.upper and memory.status[0] != 1:
        # The convoluted sizing equation below does the following: 
        # (a) Purchases a dynamic number of shares based on percent of portfolio value
        # (b) Uses dynamic hedge ratio to purchase approx. $$ equivilent number independent asset
        # REQUIRES FURTHER REVIEW TO ENSURE BEHAVIOR IS APPROPRIATE
        size[0] = -(outlay / c.close[c.i - 1, c.from_col + 1])[0] * (yhat / c.close[c.i - 1, c.from_col])
        size[1] = (outlay / c.close[c.i - 1, c.from_col + 1])[0]
        # The size notes below are the old simpler way implement order sizing.... 
        # size[0] = -params.order_size
        # size[1] = params.order_size
        c.call_seq_now[0] = 0
        c.call_seq_now[1] = 1
        memory.status[0] = 1
        
    elif memory.zscore[c.i - 1] < params.lower and memory.status[0] != 2:
        size[0] = (outlay / c.close[c.i - 1, c.from_col + 1])[0] * (yhat / c.close[c.i - 1, c.from_col])
        size[1] = -(outlay / c.close[c.i - 1, c.from_col + 1])[0]
        # size[0] = params.order_size
        # size[1] = -params.order_size
        c.call_seq_now[0] = 1  # execute the second order first to release funds early
        c.call_seq_now[1] = 0
        memory.status[0] = 2

    elif memory.status[0] == 1:
        if np.abs(memory.zscore[c.i - 1]) < params.exit:
            size[0] = 0.0
            size[1] = 0.0
            c.call_seq_now[0] = 1
            c.call_seq_now[1] = 0
            memory.status[0] = 0
        
    elif memory.status[0] == 2:
        if np.abs(memory.zscore[c.i - 1]) < params.exit:
            size[0] = 0.0
            size[1] = 0.0
            c.call_seq_now[0] = 0
            c.call_seq_now[1] = 1
            memory.status[0] = 0
        
    else:
        size[0] = np.nan
        size[1] = np.nan
        
    return (size,)

@njit
def order_func_nb(c, size, price, commperc, slippage):
    """Place an order (= element within group and row)."""
    # Get column index within group (if group starts at column 58 and current column is 59, 
    # the column within group is 1, which can be used to get size)
    group_col = c.col - c.from_col
    return portfolio_nb.order_nb(
        size=size[group_col], 
        price=price[c.i, c.col],
        size_type=SizeType.TargetAmount,
        fees=commperc,
        slippage=slippage,
    )


