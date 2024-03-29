import numpy as np
from numba import njit
from collections import namedtuple
from vectorbt.base.reshape_fns import flex_select_auto_nb
from .components.models import KF
from .components.preprocessors import (
    log_price_transform, log_return_transform, cummulative_return_transform,
)

Memory = namedtuple("Memory", ('theta', 'Ct', 'Rt', 'spread', 'signal', 'status', 'mtm'))
Params = namedtuple("Params", ('entry', 'exit', 'delta', 'vt', 'order_size', 'burnin'))
Transformations = namedtuple("Transformations", ("log", "logret", "cumlog"))


@njit
def pre_group_func_nb(c, _entry, _exit, _delta, _vt, _order_size, _burnin):
    """Prepare the current group (= pair of columns)."""

    assert c.group_len == 2

    spread = np.full(c.target_shape[0], np.nan, dtype=np.float_)
    signal = np.full(c.target_shape[0], np.nan, dtype=np.float_)


    # Calculate the transformation upfront to reference (if needed) later
    arr = c.close[:, :c.from_col+c.group_len] # This will pull out the closes as an Nx2 array
    log = log_price_transform(arr)
    logret = log_return_transform(arr)
    cumlog = cummulative_return_transform(arr)

    transformations = Transformations(log, logret, cumlog) # Store the transformations here

    # Add matrix names and descriptions as notes for posterior review
    theta = np.full(2, 0, dtype=np.float_)          # 2x1 matrix representing beta, intercept from filter
    Rt = np.full((2,2), np.nan, dtype=np.float_)    # Generates matrix of [[0,0],[0,0]]
    Ct = np.full((2,2), np.nan, dtype=np.float_)    # C matrix is correctly implemented here

    status = np.full(1, 0, dtype=np.int_)
    mtm = np.full(2, 0, dtype=np.float_)

    memory = Memory(theta, Rt, Ct, spread, signal, status, mtm)
    
    # Treat each param as an array with value per group, and select the combination of params for this group
    entry = flex_select_auto_nb(np.asarray(_entry), 0, c.group, True)
    exit = flex_select_auto_nb(np.asarray(_exit), 0, c.group, True)
    order_size = flex_select_auto_nb(np.asarray(_order_size), 0, c.group, True)
    burnin = flex_select_auto_nb(np.asarray(_burnin), 0, c.group, True) # Burnin for LQE to obtain accurate estimates

    vt = flex_select_auto_nb(np.asarray(_vt), 0, c.group, True)
    # When using wt it must be multipled by I to return 2x2 perturbance matrix
    delta = flex_select_auto_nb(np.asarray(_delta), 0, c.group, True)
    
    params = Params(entry, exit, delta, vt, order_size, burnin)
    
    size = np.empty(c.group_len, dtype=np.float_)

    return (memory, params, size, transformations)
    

@njit
def pre_segment_func_nb(c, memory, params, size, transformations, transform, hedge):
    """Prepare the current segment (= row within group)."""

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

        Rt, Ct, theta, e, Qt = KF(X, y, memory.Rt, memory.Ct, memory.theta, params.delta, params.vt)

        memory.spread[c.i] = e[0]
        memory.signal[c.i] = np.sqrt(Qt).flatten()[0] # Calculate STD of observation
        memory.theta[0:2] = theta
        memory.Rt[0], memory.Rt[1] = Rt[0], Rt[1]
        memory.Ct[0], memory.Ct[1] = Ct[0], Ct[1]

    # This statement ensures that no extraneous computing is done for signal calc prior 
    # to the burn-in period
    if c.i < params.burnin - 1:
        size[0] = np.nan  # size of nan means no order
        size[1] = np.nan
        return (size,)

    if c.i > params.burnin:

        outlay = c.last_value[c.group] * params.order_size

        # if memory.status[0] != 0 and memory.status[0] != 3:
        #     # Evaluate the net mark-to-market gain/loss
        #     marktomarket = c.last_value[c.group] - c.second_last_value[c.group]
        #     if not memory.mtm[1]:
        #         memory.mtm[1] = c.second_last_value[c.group]
        #     memory.mtm[0] = memory.mtm[0] + marktomarket
        #     pnl_pct = memory.mtm[0] / memory.mtm[1]

        if memory.spread[c.i - 1] > (params.entry * memory.signal[c.i - 1]) and not memory.status[0]:
            if hedge == "dollar":
                size[0] = outlay / c.close[c.i - 1, c.from_col] # X asset
                size[1] = -outlay / c.close[c.i - 1, c.from_col + 1] # y asset            
            elif hedge == "beta":
                if theta[0] < 1:
                    size[0] = (outlay * theta[0]) / c.close[c.i - 1, c.from_col]
                    size[1] = -outlay / c.close[c.i - 1, c.from_col + 1]
                if theta[0] > 1:
                    size[0] = outlay / c.close[c.i - 1, c.from_col]
                    size[1] = -(outlay / theta[0]) / c.close[c.i - 1, c.from_col + 1]
            c.call_seq_now[0] = 1 # Execute short sale first
            c.call_seq_now[1] = 0 # Use funds to purchase long side
            memory.status[0] = 1

        elif memory.spread[c.i - 1] < (-params.entry * memory.signal[c.i - 1]) and not memory.status[0]:
            if hedge == "dollar":
                size[0] = -outlay / c.close[c.i - 1, c.from_col] # X asset
                size[1] = outlay / c.close[c.i - 1, c.from_col + 1] # y asset
            elif hedge == "beta":
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
            # if pnl_pct < -0.05:
            #     size[0] = 0
            #     size[1] = 0
            #     c.call_seq_now[0] = 1
            #     c.call_seq_now[1] = 0
            #     memory.status[0] = 3
            #     memory.mtm[0] = 0
            #     memory.mtm[1] = 0
            if np.abs(memory.spread[c.i - 1]) < (params.exit * memory.signal[c.i - 1]):
                size[0] = 0
                size[1] = 0
                c.call_seq_now[0] = 0
                c.call_seq_now[1] = 1
                memory.status[0] = 0
                memory.mtm[0] = 0
                memory.mtm[1] = 0
            
        elif memory.status[0] == 2:
            # if pnl_pct < -0.05:
            #     size[0] = 0
            #     size[1] = 0
            #     c.call_seq_now[0] = 1
            #     c.call_seq_now[1] = 0
            #     memory.status[0] = 3
            #     memory.mtm[0] = 0
            #     memory.mtm[1] = 0
            if np.abs(memory.spread[c.i - 1]) < (params.exit * memory.signal[c.i - 1]):
                size[0] = 0
                size[1] = 0
                c.call_seq_now[0] = 1
                c.call_seq_now[1] = 0
                memory.status[0] = 0
                memory.mtm[0] = 0
                memory.mtm[1] = 0
                
        # If stop loss triggered do not trade till next mean reversion cycle
        elif memory.status[0] == 3:
            if np.abs(memory.spread[c.i - 1]) < (params.exit * memory.signal[c.i - 1]): 
                memory.status[0] = 0
            
        else:
            size[0] = np.nan
            size[1] = np.nan
        
    return (size,)
