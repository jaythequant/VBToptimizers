from numba import njit
from vectorbt.portfolio import nb as portfolio_nb
from vectorbt.portfolio.enums import SizeType


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
        log=True,
    )
