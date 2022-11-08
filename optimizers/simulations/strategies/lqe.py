import numpy as np
from numba import njit
from collections import namedtuple
from vectorbt.base.reshape_fns import flex_select_auto_nb
from .components.models import KF, discretized_OU
from .components.preprocessors import (
    log_price_transform, log_return_transform, cummulative_return_transform,
)

Memory = namedtuple("Memory", ('theta', 'Ct', 'Rt', 'spread', 'zscore', 'status', 'mtm', 'ts'))       
Params = namedtuple("Params", ('period', 'upper', 'lower', 'exit', 'delta', 'vt', 'order_size', 'burnin'))
Transformations = namedtuple("Transformations", ("log", "logret", "cumlog"))

@njit
def seed_filter_params(seed, delta, vt):
    """Use historic seed data to pre-calculate initial values for theta, R, and C matrix"""
    theta = np.full(2, 0, dtype=np.float_)
    Rt = np.full((2,2), np.nan, dtype=np.float_)
    Ct = np.full((2,2), np.nan, dtype=np.float_)
    for idx in range(0, seed.shape[0]):
        bar = seed[idx,:]
        Rt, Ct, theta, _, _ = KF(bar[0], bar[1], Rt, Ct, theta, delta, vt)
        theta[0:2] = theta
        Rt[0], Rt[1] = Rt[0], Rt[1]
        Ct[0], Ct[1] = Ct[0], Ct[1]
    return theta, Rt, Ct

@njit
def lqe_pre_group_func_nb(c, _period, _upper, _lower, _exit, _delta, _vt, _order_size, _burnin, _seed):
    """Prepare the current group (= pair of columns)."""

    assert c.group_len == 2

    spread = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    zscore = np.full(c.target_shape[0], np.nan, dtype=np.float_)

    # Calculate the transformation upfront to reference (if needed) later
    arr = c.close[:, :c.from_col+c.group_len] # This will pull out the closes as an Nx2 array
    log = log_price_transform(arr)
    logret = log_return_transform(arr)
    cumlog = cummulative_return_transform(arr)

    transformations = Transformations(log, logret, cumlog) # Store the transformations here

    if _seed.sum() > 0:
        theta, Rt, Ct = seed_filter_params(_seed, _delta, _vt)
    else:
        theta = np.full(2, 0, dtype=np.float_)
        Rt = np.full((2,2), np.nan, dtype=np.float_)
        Ct = np.full((2,2), np.nan, dtype=np.float_)

    status = np.full(1, 0, dtype=np.int_)
    mtm = np.full(2, 0, dtype=np.float_)
    ts = np.full(2, 0, dtype=np.float_) # Container for timestamping trades based on index

    memory = Memory(theta, Rt, Ct, spread, zscore, status, mtm, ts)
    
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
def lqe_pre_segment_func_nb(c, memory, params, size, transformations, transform, hedge, standardization):
    """Prepare the current segment (= row within group)."""
    
    # In state space implentation a burn-in period is needed
    # Note that use of rolling statistics introduces a (potentially) un-needed parameter to the hypertuning process...

    if c.i > 0: # Ignore the first element as it is always zero when transformation is applied
        if not transform or transform == "default":
            X = c.close[c.i, c.from_col]         
            y = c.close[c.i, c.from_col + 1]
        elif transform == "cumlog":
            X = transformations.cumlog[c.i, c.from_col]
            y = transformations.cumlog[c.i, c.from_col + 1]
        elif transform == "logret":
            X = transformations.logret[c.i, c.from_col]
            y = transformations.logret[c.i, c.from_col + 1]
        elif transform == "log":
            X = transformations.log[c.i, c.from_col]
            y = transformations.log[c.i, c.from_col + 1]

        Rt, Ct, theta, e, _ = KF(X, y, memory.Rt, memory.Ct, memory.theta, params.delta, params.vt)

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

    if c.i > params.burnin - 1:

        # rolling statistics are calculated using a window (=period) of error terms
        # This window can be specified as a slice
        window_slice = slice(max(0, c.i + 1 - params.period), c.i + 1)

        if standardization == 'zscore':
            # Calculate rolling statistics and normalized z-score
            spread_mean = np.mean(memory.spread[window_slice])
            spread_std = np.std(memory.spread[window_slice])
            memory.zscore[c.i] = (memory.spread[c.i] - spread_mean) / spread_std
        elif standardization == 'sscore':
            # Calculate rolling s-score as presented in Avallenda et al. 2008
            memory.zscore[c.i] = discretized_OU(memory.spread[window_slice])
        elif standardization == 'sscorealt':
            # Experimentatal s-score using final residual for s-score calc (despite Avallenda saying it is unneccesary)
            memory.zscore[c.i] = discretized_OU(memory.spread[window_slice], alternative_calc=True)

        outlay = c.last_value[c.group] * params.order_size
        nigo_trade = np.abs(memory.zscore[c.i - 1]) > 6

        # A crude mark-to-market calculation
        # if memory.status[0] != 0 and memory.status[0] != 3:
        #     # Evaluate the net mark-to-market gain/loss
        #     marktomarket = c.last_value[c.group] - c.second_last_value[c.group]
        #     if not memory.mtm[1]:
        #         last_prices = np.asarray([c.close[c.i - 1, c.from_col], c.close[c.i - 1, c.from_col + 1]])
        #         init_vals =  last_prices * size
        #         memory.mtm[1] = np.sum(np.abs(init_vals))
        #         # memory.mtm[1] = c.second_last_value[c.group]
        #     memory.mtm[0] = memory.mtm[0] + marktomarket
        #     pnl_pct = memory.mtm[0] / memory.mtm[1]

        # Check if any bound is crossed
        # Since zscore is calculated using close, use zscore of the previous step
        # This way we are executing signals defined at the previous bar
        if memory.zscore[c.i - 1] > params.upper and not memory.status[0] and theta[0] > 0.05 and not nigo_trade:
            memory.ts[0] = c.i
            if hedge == "dollar":
                size[0] = outlay / c.close[c.i - 1, c.from_col] # X asset
                size[1] = -outlay / c.close[c.i - 1, c.from_col + 1] # y asset
            elif hedge == "beta":
                # In the event that our beta hedge is greater than 1, we would 
                # be opening positions much larger than our target percentage size.
                # In response, open identically profiled positions, but use beta
                # to scale our contra-asset down by moving to the numerator of the Y asset
                if np.abs(theta[0]) < 1:
                    size[0] = (outlay * theta[0]) / c.close[c.i - 1, c.from_col]
                    size[1] = -outlay / c.close[c.i - 1, c.from_col + 1]
                elif np.abs(theta[0]) >= 1:
                    size[0] = outlay / c.close[c.i - 1, c.from_col]
                    size[1] = -(outlay / theta[0]) / c.close[c.i - 1, c.from_col + 1]
            elif hedge == "betaunit":
                # Delta-based hedge strategy. We assume that delta shares y == beta * delta shares x
                delta_y = outlay / c.close[c.i - 1, c.from_col + 1] # n *shares* of y
                size[0] = delta_y * theta[0] # beta n *shares* of x
                size[1] = -delta_y
            c.call_seq_now[0] = 1 # Execute short sale first
            c.call_seq_now[1] = 0 # Use funds to purchase long side
            memory.status[0] = 1
                
        # Note that x_t = c.close[c.i - 1, c.from_col]
        # and y_t = c.close[c.i - 1, c.from_col + 1]

        elif memory.zscore[c.i - 1] < params.lower and not memory.status[0] and theta[0] > 0.05 and not nigo_trade:
            memory.ts[0] = c.i
            if hedge == "dollar":
                size[0] = -outlay / c.close[c.i - 1, c.from_col] # X asset
                size[1] = outlay / c.close[c.i - 1, c.from_col + 1] # y asset
            elif hedge == "beta":
                if np.abs(theta[0]) < 1:
                    size[0] = -(outlay * theta[0]) / c.close[c.i - 1, c.from_col]
                    size[1] = outlay / c.close[c.i - 1, c.from_col + 1]
                elif np.abs(theta[0]) >= 1:
                    size[0] = -outlay / c.close[c.i - 1, c.from_col]
                    size[1] = (outlay / theta[0]) / c.close[c.i - 1, c.from_col + 1]
            elif hedge == "betaunit":
                delta_y = outlay / c.close[c.i - 1, c.from_col + 1] # n *shares* of y
                size[0] = -(delta_y * theta[0]) # beta n *shares* of x
                size[1] = delta_y
            c.call_seq_now[0] = 0  # execute the second order first to release funds early
            c.call_seq_now[1] = 1
            memory.status[0] = 2

        elif memory.status[0] == 1:
            if memory.zscore[c.i - 1] < params.exit or c.i == memory.ts[0]+336:
                size[0] = 0
                size[1] = 0
                c.call_seq_now[0] = 0
                c.call_seq_now[1] = 1
                memory.status[0] = 0
                memory.mtm[0] = 0
                memory.mtm[1] = 0
            
        elif memory.status[0] == 2:
            if memory.zscore[c.i - 1] > params.exit or c.i == memory.ts[0]+336:
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
