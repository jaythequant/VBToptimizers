from setup.lqe_setup import * # Import a state space model setup implementing Kalman Filtering
import vectorbt as vbt


def simulate_from_order_func(
    close_data, open_data, period, upper, lower, exits, burnin=500, delta=1e-5, vt=1, 
    mode="Kalman", cash=100_000, commission=0.0008, slippage=0.0010, order_size=0.10, 
    freq="d"
):
    """Simulate pairs trading strategy with multiple signal strategy optionality"""
    return vbt.Portfolio.from_order_func(
        close_data,
        order_func_nb, 
        open_data.values, commission, slippage,  # *args for order_func_nb
        pre_group_func_nb=pre_group_func_nb, 
        pre_group_args=(
            period, 
            upper, 
            lower, 
            exits, 
            delta, 
            vt, 
            order_size, 
            burnin,
        ),
        pre_segment_func_nb=pre_segment_func_nb, 
        pre_segment_args=(mode,),
        fill_pos_record=False,  # a bit faster
        init_cash=cash,
        cash_sharing=True, 
        group_by=True,
        freq=freq
    )
