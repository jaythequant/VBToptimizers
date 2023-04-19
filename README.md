# VBToptimizers
Use genetic programming to optimize simple pairs trading strategies. (A work in progress)

A super fast, super optimized platform for backtesting pairs trading strategies and conducting hyperparameter tuning via genetic programming. 
This has has been a passion project of mine for some time and I wanted to share the code with everyone as I think it's a lovely solution to
an extremely complex search problem.

For building the SQL database to feed the backtesting algos, please review KuCoin-Cli, my open-source data science focused API wrapper.
Additionally, the backtesting back-end is running on a lovely backtesting framework called VectorBT which itself is written on top of 
numba-accelerated numpy. It may be difficult to tweak or understand the code without a thorough understanding of these these libraries.

Once life slows down a bit, I intend to thoroughly document the project and refactor the code to an OOP structure.
