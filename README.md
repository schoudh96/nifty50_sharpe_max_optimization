This project aims to build an optimal asset allocation strategy on the universe of NIFTY 50 and help beat the standard market-capitalization allocation and has the following features:

1. PCA-based Factor Analysis to denoise shared covariance structure, cutting OOS Frobenius-norm error vs. realized covariance by 20% through hyperparameter tuning
2. Identification co-moving stock clusters via K-Means and application of Nested Clustered Optimization per cluster running max-sharpe strategy for the optimization and a final cluster/sleeve level NCO for final allocations.
3. Framework is config-driven and implemented using CVXPY and Python

