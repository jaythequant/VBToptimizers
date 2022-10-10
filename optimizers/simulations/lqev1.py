import numpy as np
from numba import njit
from collections import namedtuple
from vectorbt.portfolio import nb as portfolio_nb
from vectorbt.base.reshape_fns import flex_select_auto_nb
from vectorbt.portfolio.enums import SizeType

Memory = namedtuple("Memory", ('theta', 'Ct', 'Rt', 'spread', 'zscore', 'status', 'mtm'))       
Params = namedtuple("Params", ('period', 'upper', 'lower', 'exit', 'delta', 'vt', 'order_size', 'burnin'))
Transformations = namedtuple("Transformations", ("cumm_x", "cumm_y", "logr_x", "logr_y", "log_x", "log_y"))


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
    
    return R, C, theta, et


@njit
def pre_group_func_nb(c, _period, _upper, _lower, _exit, _delta, _vt, _order_size, _burnin):
    """Prepare the current group (= pair of columns)."""

    assert c.group_len == 2

    spread = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    zscore = np.full(c.target_shape[0], np.nan, dtype=np.float_)

    # Calculate the transformation upfront to reference (if needed) later
    x_arr = c.close[:, c.from_col]
    y_arr = c.close[:, c.from_col + 1]
    cumm_x = np.full(x_arr.shape, 0, dtype=np.float_)
    cumm_y = np.full(y_arr.shape, 0, dtype=np.float_)
    logr_x = np.full(x_arr.shape, 0, dtype=np.float_)
    logr_y = np.full(y_arr.shape, 0, dtype=np.float_)
    cumm_x[1:] = np.log1p((x_arr[1:] - x_arr[:-1]) / x_arr[:-1]).cumsum()
    cumm_y[1:] = np.log1p((y_arr[1:] - y_arr[:-1]) / y_arr[:-1]).cumsum()
    logr_x[1:] = np.log1p((x_arr[1:] - x_arr[:-1]) / x_arr[:-1])
    logr_y[1:] = np.log1p((y_arr[1:] - y_arr[:-1]) / y_arr[:-1])
    log_x = np.log(x_arr)
    log_y = np.log(y_arr)

    transformations = Transformations(cumm_x, cumm_y, logr_x, logr_y, log_x, log_y) # Store the transformations here

    # Add matrix names and descriptions as notes for posterior review
    theta = np.full(2, 0, dtype=np.float_)          # 2x1 matrix representing beta, intercept from filter
    Rt = np.full((2,2), np.nan, dtype=np.float_)    # Generates matrix of [[0,0],[0,0]]
    Ct = np.full((2,2), np.nan, dtype=np.float_)    # C matrix is correctly implemented here

    status = np.full(1, 0, dtype=np.int_)
    mtm = np.full(2, 0, dtype=np.float_)

    memory = Memory(theta, Rt, Ct, spread, zscore, status, mtm)
    
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

    return (memory, params, size, transformations)
    

@njit
def pre_segment_func_nb(c, memory, params, size, transformations, mode, hedge):
    """Prepare the current segment (= row within group)."""
    
    # In state space implentation a burn-in period is needed
    # Note that use of rolling statistics introduces a (potentially) un-needed parameter to the hypertuning process...
    
    # z-score is calculated using a window (=period) of error terms
    # This window can be specified as a slice
    window_slice = slice(max(0, c.i + 1 - params.period), c.i + 1)

    if c.i > 0: # Ignore the first element as it is always zero when transformation is applied
        if mode == "default":
            X = c.close[c.i, c.from_col]         
            y = c.close[c.i, c.from_col + 1]
        if mode == "cummlog":
            X = transformations.cumm_x[c.i]
            y = transformations.cumm_y[c.i]
        if mode == "logret":
            X = transformations.logr_x[c.i]
            y = transformations.logr_y[c.i]
        if mode == "log":
            X = transformations.log_x[c.i]
            y = transformations.log_y[c.i]

        Rt, Ct, theta, e = kf_nb(X, y, memory.Rt, memory.Ct, memory.theta, params.delta, params.vt)

        memory.spread[c.i] = e[0]
        memory.theta[0:2] = theta
        memory.Rt[0], memory.Rt[1] = Rt[0], Rt[1]
        memory.Ct[0], memory.Ct[1] = Ct[0], Ct[1]

        # This statement ensures that no extraneous computing is done for signal calc prior 
        # to the burn-in period
    if c.i < params.burnin - 1 and c.i < params.period - 1:
        size[0] = np.nan  # size of nan means no order
        size[1] = np.nan
        return (size,)

    if c.i > params.burnin + params.period - 1:

        # Calculate rolling statistics and normalized z-score
        spread_mean = np.mean(memory.spread[window_slice])
        spread_std = np.std(memory.spread[window_slice])
        memory.zscore[c.i] = (memory.spread[c.i] - spread_mean) / spread_std

        outlay = c.last_value[c.group] * params.order_size

        # A crude mark-to-market calculation
        if memory.status[0] != 0 and memory.status[0] != 3:
            # Evaluate the net mark-to-market gain/loss
            marktomarket = c.last_value[c.group] - c.second_last_value[c.group]
            if not memory.mtm[1]:
                last_prices = np.asarray([c.close[c.i - 1, c.from_col], c.close[c.i - 1, c.from_col + 1]])
                init_vals =  last_prices * size
                memory.mtm[1] = np.sum(np.abs(init_vals))
                # memory.mtm[1] = c.second_last_value[c.group]
            memory.mtm[0] = memory.mtm[0] + marktomarket
            pnl_pct = memory.mtm[0] / memory.mtm[1]

        # Check if any bound is crossed
        # Since zscore is calculated using close, use zscore of the previous step
        # This way we are executing signals defined at the previous bar
        if memory.zscore[c.i - 1] > params.upper and not memory.status[0]:
            # if memory.zscore[c.i - 1] < params.upper + 1:
                if hedge == "dollar":
                    size[0] = outlay / c.close[c.i - 1, c.from_col] # X asset
                    size[1] = -outlay / c.close[c.i - 1, c.from_col + 1] # y asset
                if hedge == "beta":
                    # In the event that our beta hedge is greater than 1, we would 
                    # be opening positions much larger than our target percentage size.
                    # In response, open identically profiled positions, but use beta
                    # to scale our contra-asset down by moving to the numerator of the Y asset
                    if theta[0] < 1:
                        size[0] = (outlay * theta[0]) / c.close[c.i - 1, c.from_col]
                        size[1] = -outlay / c.close[c.i - 1, c.from_col + 1]
                    if theta[0] > 1:
                        size[0] = outlay / c.close[c.i - 1, c.from_col]
                        size[1] = -(outlay / theta[0]) / c.close[c.i - 1, c.from_col + 1]
                c.call_seq_now[0] = 1 # Execute short sale first
                c.call_seq_now[1] = 0 # Use funds to purchase long side
                memory.status[0] = 1
                
        # Note that x_t = c.close[c.i - 1, c.from_col]
        # and y_t = c.close[c.i - 1, c.from_col + 1]

        elif memory.zscore[c.i - 1] < params.lower and not memory.status[0]:
            # if memory.zscore[c.i - 1] > params.lower - 1:
                if hedge == "dollar":
                    size[0] = -outlay / c.close[c.i - 1, c.from_col] # X asset
                    size[1] = outlay / c.close[c.i - 1, c.from_col + 1] # y asset
                if hedge == "beta":
                    if theta[0] < 1:
                        size[0] = -(outlay * theta[0]) / c.close[c.i - 1, c.from_col]
                        size[1] = outlay / c.close[c.i - 1, c.from_col + 1]
                    if theta[0] > 1:
                        size[0] = -outlay / c.close[c.i - 1, c.from_col]
                        size[1] = (outlay / theta[0]) / c.close[c.i - 1, c.from_col + 1]
                c.call_seq_now[0] = 0  # execute the second order first to release funds early
                c.call_seq_now[1] = 1  
                memory.status[0] = 2

        elif memory.status[0] == 1:
            if np.abs(memory.zscore[c.i - 1]) < params.exit:
                size[0] = 0
                size[1] = 0
                c.call_seq_now[0] = 0
                c.call_seq_now[1] = 1
                memory.status[0] = 0
                memory.mtm[0] = 0
                memory.mtm[1] = 0
            
        elif memory.status[0] == 2:
            if np.abs(memory.zscore[c.i - 1]) < params.exit:
                size[0] = 0
                size[1] = 0
                c.call_seq_now[0] = 1
                c.call_seq_now[1] = 0
                memory.status[0] = 0
                memory.mtm[0] = 0
                memory.mtm[1] = 0

        # If a trade stops out then temporarily stop looking for trades till mean reversion occurs
        # elif memory.status[0] == 3:
        #     if np.abs(memory.zscore[c.i - 1]) < params.exit: 
        #         memory.status[0] = 0
            
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
        lock_cash=False,
    )
