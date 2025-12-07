"""
Median unbiased estimators for lambda_g and lambda_z.
Direct port of median.unbiased.estimator.stage1.R and median.unbiased.estimator.stage2.R
"""
from __future__ import annotations

import numpy as np

# Values from Table 3 in Stock and Watson (1998)
VAL_EW = np.array([
    0.426, 0.476, 0.516, 0.661, 0.826, 1.111,
    1.419, 1.762, 2.355, 2.91,  3.413, 3.868, 4.925,
    5.684, 6.670, 7.690, 8.477, 9.191, 10.693, 12.024,
    13.089, 14.440, 16.191, 17.332, 18.699, 20.464,
    21.667, 23.851, 25.538, 26.762, 27.874
])

VAL_MW = np.array([
    0.689, 0.757, 0.806, 1.015, 1.234, 1.632,
    2.018, 2.390, 3.081, 3.699, 4.222, 4.776, 5.767,
    6.586, 7.703, 8.683, 9.467, 10.101, 11.639, 13.039,
    13.900, 15.214, 16.806, 18.330, 19.020, 20.562,
    21.837, 24.350, 26.248, 27.089, 27.758
])

VAL_QL = np.array([
    3.198, 3.416, 3.594, 4.106, 4.848, 5.689,
    6.682, 7.626, 9.16,  10.66, 11.841, 13.098, 15.451,
    17.094, 19.423, 21.682, 23.342, 24.920, 28.174, 30.736,
    33.313, 36.109, 39.673, 41.955, 45.056, 48.647, 50.983,
    55.514, 59.278, 61.311, 64.016
])


def _interpolate_lambda(stat: float, val_table: np.ndarray) -> float:
    """Interpolate lambda from table values using Stock-Watson procedure."""
    if stat <= val_table[0]:
        return 0.0
    for i in range(len(val_table) - 1):
        if val_table[i] < stat <= val_table[i + 1]:
            return i + (stat - val_table[i]) / (val_table[i + 1] - val_table[i])
    return np.nan


def median_unbiased_estimator_stage1(series: np.ndarray) -> float:
    """
    Direct port of median.unbiased.estimator.stage1.R
    
    Implements median unbiased estimation of lambda_g following Stock and Watson (1998).
    """
    t_end = len(series)
    y = 400 * np.diff(series)  # Annualized growth rate
    
    stat = np.zeros(t_end - 2 * 4)
    
    for i in range(4, t_end - 4):
        # Build regressor matrix with structural break
        xr = np.column_stack([
            np.ones(t_end - 1),
            np.concatenate([np.zeros(i), np.ones(t_end - i - 1)])
        ])
        
        # OLS
        xi = np.linalg.inv(xr.T @ xr)
        b = np.linalg.solve(xr.T @ xr, xr.T @ y)
        s3 = np.sum((y - xr @ b) ** 2) / (t_end - 2 - 1)
        
        # t-statistic for break coefficient
        stat[i - 4] = b[1] / np.sqrt(s3 * xi[1, 1])
    
    # Calculate test statistics
    ew = np.log(np.mean(np.exp(stat ** 2 / 2)))
    mw = np.sum(stat ** 2) / len(stat)
    qlr = np.max(stat ** 2)
    
    # Interpolate lambda values
    lame = _interpolate_lambda(ew, VAL_EW)
    lamm = _interpolate_lambda(mw, VAL_MW)
    lamq = _interpolate_lambda(qlr, VAL_QL)
    
    if np.isnan(lame) or np.isnan(lamm) or np.isnan(lamq):
        print("Warning: At least one statistic has an NA value.")
    
    return lame / (t_end - 1)


def median_unbiased_estimator_stage2(
    y: np.ndarray, 
    x: np.ndarray, 
    kappa_vec: np.ndarray
) -> float:
    """
    Direct port of median.unbiased.estimator.stage2.R
    
    Implements median unbiased estimation of lambda_z following Stock and Watson (1998).
    """
    t_end = x.shape[0]
    stat = np.zeros(t_end - 2 * 4 + 1)
    
    # Weight matrix
    w = np.diag(1 / (kappa_vec ** 2))
    
    for i in range(4, t_end - 3):
        # Build regressor matrix with structural break
        xr = np.column_stack([
            x,
            np.concatenate([np.zeros(i), np.ones(t_end - i)])
        ])
        
        # Weighted OLS
        xi = np.linalg.inv(xr.T @ w @ xr)
        b = np.linalg.solve(xr.T @ w @ xr, xr.T @ w @ y)
        s3 = np.sum(w @ (y - xr @ b) ** 2) / (np.sum(np.diag(w)) - xr.shape[1])
        
        # t-statistic for break coefficient
        stat[i - 4] = b[-1] / np.sqrt(s3 * xi[-1, -1])
    
    # Calculate test statistics
    ew = np.log(np.mean(np.exp(stat ** 2 / 2)))
    mw = np.mean(stat ** 2)
    qlr = np.max(stat ** 2)
    
    # Interpolate lambda values
    lame = _interpolate_lambda(ew, VAL_EW)
    lamm = _interpolate_lambda(mw, VAL_MW)
    lamq = _interpolate_lambda(qlr, VAL_QL)
    
    if np.isnan(lame) or np.isnan(lamm) or np.isnan(lamq):
        print("Warning: At least one statistic has an NA value.")
    
    return lame / t_end


