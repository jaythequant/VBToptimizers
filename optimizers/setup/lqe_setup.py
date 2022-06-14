import numpy as np
from numba import njit
from collections import namedtuple

from vectorbt.portfolio import nb as portfolio_nb
from vectorbt.base.reshape_fns import flex_select_auto_nb
from vectorbt.portfolio.enums import SizeType

Memory = namedtuple("Memory", ('theta', 'Ct', 'Rt', 'spread', 'zscore', 'status'))       
Params = namedtuple("Params", ('period', 'upper', 'lower', 'exit', 'delta', 'vt', 'order_size', 'burnin'))


@njit
def kf_nb(X, y, R, C, theta, delta=1e-5, vt=1):
    """Numba compiled kalman filter implentation"""
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
    
    return R, C, theta, et, yhat


@njit
def pre_group_func_nb(c, _period, _upper, _lower, _exit, _delta, _vt, _order_size, _burnin):
    """Prepare the current group (= pair of columns)."""

    assert c.group_len == 2

    spread = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    zscore = np.full(c.target_shape[0], np.nan, dtype=np.float_)

    # Add matrix names and descriptions as notes for posterior review
    theta = np.full(2, 0, dtype=np.float_)          # 2x1 matrix representing beta, intercept from filter
    Rt = np.full((2,2), np.nan, dtype=np.float_)    # Generates matrix of [[0,0],[0,0]]
    Ct = np.full((2,2), np.nan, dtype=np.float_)    # C matrix is correctly implemented here

    status = np.full(1, 0, dtype=np.int_)

    memory = Memory(theta, Rt, Ct, spread, zscore, status)
    
    # Treat each param as an array with value per group, and select the combination of params for this group
    period = flex_select_auto_nb(np.asarray(_period), 0, c.group, True)
    upper = flex_select_auto_nb(np.asarray(_upper), 0, c.group, True)
    lower = flex_select_auto_nb(np.asarray(_lower), 0, c.group, True)
    exit = flex_select_auto_nb(np.asarray(_exit), 0, c.group, True)
    order_size = flex_select_auto_nb(np.asarray(_order_size), 0, c.group, True)
    burnin = flex_select_auto_nb(np.asarray(_burnin), 0, c.group, True) # Burnin for LQE to obtain accurate estimates

    vt = flex_select_auto_nb(np.asarray(_vt), 0, c.group, True)
    # When using wt it must be multipled by I to return 2x2 perturbance matrix
    delta = flex_select_auto_nb(np.asarray(_delta), 0, c.group, True)
    
    params = Params(period, upper, lower, exit, delta, vt, order_size, burnin)
    
    size = np.empty(c.group_len, dtype=np.float_)

    return (memory, params, size)
    

@njit
def pre_segment_func_nb(c, memory, params, size, mode):
    """Prepare the current segment (= row within group)."""
    
    # In state space implentation a burn-in period is needed
    # Note that use of rolling statistics introduces a (potentially) un-needed parameter to the hypertuning process...
    
    # z-score is calculated using a window (=period) of error terms
    # This window can be specified as a slice
    window_slice = slice(max(0, c.i + 1 - params.period), c.i + 1)

    # Ultimately I will remove the mode flag as I am compartmentalizing models into different setup files
    if mode == "Kalman":
        X = c.close[c.i, c.from_col]                     
        y = c.close[c.i, c.from_col + 1]            

        Rt, Ct, theta, e, yhat = kf_nb(X, y, memory.Rt, memory.Ct, memory.theta, params.delta, params.vt)

        # Update all memory variables for next iteration of filter process
        # Also isolate yhat float val from array (neccesary due to immutability of namedtuples)
        memory.spread[c.i] = e[0]
        memory.theta[0:2] = theta
        memory.Rt[0], memory.Rt[1] = Rt[0], Rt[1]
        memory.Ct[0], memory.Ct[1] = Ct[0], Ct[1]
        yhat = yhat[0]

    # This statement ensures that no extraneous computing is done for signal calc prior 
    # to the burn-in period
    if c.i < params.burnin - 1 and c.i < params.period - 1:
        size[0] = np.nan  # size of nan means no order
        size[1] = np.nan
        return (size,)

    if c.i > params.burnin - 1 and c.i > params.period - 1:

        # Calculate rolling statistics and normalized z-score
        spread_mean = np.mean(memory.spread[window_slice])
        spread_std = np.std(memory.spread[window_slice])
        memory.zscore[c.i] = (memory.spread[c.i] - spread_mean) / spread_std

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
