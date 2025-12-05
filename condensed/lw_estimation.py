#!/usr/bin/env python3
"""
Laubach-Williams (2003) Natural Rate Estimation - Single File Version

This is a consolidated version of the LW Python port that combines all modules
into a single runnable script. It replicates the NY Fed's r* estimation using
Kalman filtering and maximum likelihood.

Usage:
    python lw_estimation.py [path_to_excel]

If no path is provided, it looks for the data file in ../data/
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from statsmodels.tsa.filters.hp_filter import hpfilter

# =============================================================================
# KALMAN FILTER AND SMOOTHER
# =============================================================================

@dataclass
class FilteredStates:
    xi_ttm1: np.ndarray
    P_ttm1: np.ndarray
    xi_tt: np.ndarray
    P_tt: np.ndarray
    prediction_error: np.ndarray
    kalman_gain: np.ndarray


@dataclass
class SmoothedStates:
    xi_tT: np.ndarray
    P_tT: np.ndarray


@dataclass
class KalmanStates:
    filtered: FilteredStates
    smoothed: SmoothedStates


def kalman_log_likelihood(
    xi_tm1tm1: np.ndarray,
    P_tm1tm1: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    kappa: np.ndarray,
    cons: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Kalman filter log-likelihood calculation."""
    t_end = y.shape[0]
    n = y.shape[1]
    
    ll_vec = np.zeros(t_end)
    ll_cum = 0.0
    prediction_error_vec = np.zeros((t_end, n))
    
    xi_tt = xi_tm1tm1.copy()
    P_tt = P_tm1tm1.copy()
    
    for t in range(t_end):
        xi_ttm1 = F @ xi_tt + cons.flatten()
        P_ttm1 = F @ P_tt @ F.T + Q
        prediction_error = y[t] - A.T @ x[t] - H.T @ xi_ttm1
        HPHR = H.T @ P_ttm1 @ H + (kappa[t] ** 2) * R
        
        sign, logdet = np.linalg.slogdet(HPHR)
        if sign <= 0:
            ll_vec[t] = -1e10
        else:
            ll_vec[t] = (
                -(n / 2) * np.log(4 * np.arctan(1))
                - 0.5 * logdet
                - 0.5 * prediction_error @ np.linalg.solve(HPHR, prediction_error)
            )
        ll_cum += ll_vec[t]
        prediction_error_vec[t] = prediction_error
        
        kalman_gain = P_ttm1 @ H @ np.linalg.inv(HPHR)
        xi_tt = xi_ttm1 + P_ttm1 @ H @ np.linalg.solve(HPHR, prediction_error)
        P_tt = P_ttm1 - P_ttm1 @ H @ np.linalg.solve(HPHR, H.T @ P_ttm1)
    
    return ll_vec, ll_cum, prediction_error_vec


def kalman_states_filtered(
    xi_tm1tm1: np.ndarray,
    P_tm1tm1: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    kappa: np.ndarray,
    cons: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> FilteredStates:
    """Kalman filter forward pass."""
    t_end = y.shape[0]
    n_state = len(xi_tm1tm1)
    n_obs = y.shape[1]
    
    xi_ttm1_arr = np.zeros((t_end, n_state))
    P_ttm1_arr = np.zeros((t_end * n_state, n_state))
    xi_tt_arr = np.zeros((t_end, n_state))
    P_tt_arr = np.zeros((t_end * n_state, n_state))
    prediction_error_arr = np.zeros((t_end, n_obs))
    kalman_gain_arr = np.zeros((t_end * n_state, n_obs))
    
    xi_tt = xi_tm1tm1.copy()
    P_tt = P_tm1tm1.copy()
    
    for t in range(t_end):
        xi_ttm1 = F @ xi_tt + cons.flatten()
        P_ttm1 = F @ P_tt @ F.T + Q
        prediction_error = y[t] - A.T @ x[t] - H.T @ xi_ttm1
        HPHR = H.T @ P_ttm1 @ H + (kappa[t] ** 2) * R
        
        xi_tt = xi_ttm1 + P_ttm1 @ H @ np.linalg.solve(HPHR, prediction_error)
        P_tt = P_ttm1 - P_ttm1 @ H @ np.linalg.solve(HPHR, H.T @ P_ttm1)
        kalman_gain = P_ttm1 @ H @ np.linalg.inv(HPHR)
        
        xi_ttm1_arr[t] = xi_ttm1
        P_ttm1_arr[t * n_state:(t + 1) * n_state, :] = P_ttm1
        xi_tt_arr[t] = xi_tt
        P_tt_arr[t * n_state:(t + 1) * n_state, :] = P_tt
        prediction_error_arr[t] = prediction_error
        kalman_gain_arr[t * n_state:(t + 1) * n_state, :] = kalman_gain
    
    return FilteredStates(
        xi_ttm1=xi_ttm1_arr, P_ttm1=P_ttm1_arr, xi_tt=xi_tt_arr,
        P_tt=P_tt_arr, prediction_error=prediction_error_arr, kalman_gain=kalman_gain_arr,
    )


def kalman_states_smoothed(filtered: FilteredStates, F: np.ndarray) -> SmoothedStates:
    """Kalman smoother backward pass."""
    t_end = filtered.xi_tt.shape[0]
    n_state = filtered.xi_tt.shape[1]
    
    xi_tT_arr = np.zeros((t_end, n_state))
    P_tT_arr = np.zeros((t_end * n_state, n_state))
    
    xi_tT_arr[t_end - 1] = filtered.xi_tt[t_end - 1]
    P_tT_arr[(t_end - 1) * n_state:t_end * n_state, :] = filtered.P_tt[(t_end - 1) * n_state:t_end * n_state, :]
    
    for t in range(t_end - 2, -1, -1):
        P_tt = filtered.P_tt[t * n_state:(t + 1) * n_state, :]
        P_tp1t = filtered.P_ttm1[(t + 1) * n_state:(t + 2) * n_state, :]
        
        try:
            J_t = P_tt @ F.T @ np.linalg.solve(P_tp1t, np.eye(n_state))
        except np.linalg.LinAlgError:
            J_t = P_tt @ F.T @ np.linalg.solve(P_tp1t + 1e-10 * np.eye(n_state), np.eye(n_state))
        
        xi_tt = filtered.xi_tt[t]
        xi_tp1t = filtered.xi_ttm1[t + 1]
        xi_tp1T = xi_tT_arr[t + 1]
        P_tp1T = P_tT_arr[(t + 1) * n_state:(t + 2) * n_state, :]
        
        xi_tT_arr[t] = xi_tt + J_t @ (xi_tp1T - xi_tp1t)
        P_tT_arr[t * n_state:(t + 1) * n_state, :] = P_tt + J_t @ (P_tp1T - P_tp1t) @ J_t.T
    
    return SmoothedStates(xi_tT=xi_tT_arr, P_tT=P_tT_arr)


def kalman_states(
    xi_tm1tm1: np.ndarray, P_tm1tm1: np.ndarray, F: np.ndarray, Q: np.ndarray,
    A: np.ndarray, H: np.ndarray, R: np.ndarray, kappa: np.ndarray,
    cons: np.ndarray, y: np.ndarray, x: np.ndarray,
) -> KalmanStates:
    """Run Kalman filter and smoother."""
    filtered = kalman_states_filtered(xi_tm1tm1, P_tm1tm1, F, Q, A, H, R, kappa, cons, y, x)
    smoothed = kalman_states_smoothed(filtered, F)
    return KalmanStates(filtered=filtered, smoothed=smoothed)


# =============================================================================
# MEDIAN UNBIASED ESTIMATORS
# =============================================================================

VAL_EW = np.array([
    0.426, 0.476, 0.516, 0.661, 0.826, 1.111, 1.419, 1.762, 2.355, 2.91,
    3.413, 3.868, 4.925, 5.684, 6.670, 7.690, 8.477, 9.191, 10.693, 12.024,
    13.089, 14.440, 16.191, 17.332, 18.699, 20.464, 21.667, 23.851, 25.538, 26.762, 27.874
])

VAL_MW = np.array([
    0.689, 0.757, 0.806, 1.015, 1.234, 1.632, 2.018, 2.390, 3.081, 3.699,
    4.222, 4.776, 5.767, 6.586, 7.703, 8.683, 9.467, 10.101, 11.639, 13.039,
    13.900, 15.214, 16.806, 18.330, 19.020, 20.562, 21.837, 24.350, 26.248, 27.089, 27.758
])

VAL_QL = np.array([
    3.198, 3.416, 3.594, 4.106, 4.848, 5.689, 6.682, 7.626, 9.16, 10.66,
    11.841, 13.098, 15.451, 17.094, 19.423, 21.682, 23.342, 24.920, 28.174, 30.736,
    33.313, 36.109, 39.673, 41.955, 45.056, 48.647, 50.983, 55.514, 59.278, 61.311, 64.016
])


def _interpolate_lambda(stat: float, val_table: np.ndarray) -> float:
    if stat <= val_table[0]:
        return 0.0
    for i in range(len(val_table) - 1):
        if val_table[i] < stat <= val_table[i + 1]:
            return i + (stat - val_table[i]) / (val_table[i + 1] - val_table[i])
    return np.nan


def median_unbiased_estimator_stage1(series: np.ndarray) -> float:
    """Median unbiased estimation of lambda_g (Stock-Watson 1998)."""
    t_end = len(series)
    y = 400 * np.diff(series)
    stat = np.zeros(t_end - 2 * 4)
    
    for i in range(4, t_end - 4):
        xr = np.column_stack([
            np.ones(t_end - 1),
            np.concatenate([np.zeros(i), np.ones(t_end - i - 1)])
        ])
        xi = np.linalg.inv(xr.T @ xr)
        b = np.linalg.solve(xr.T @ xr, xr.T @ y)
        s3 = np.sum((y - xr @ b) ** 2) / (t_end - 2 - 1)
        stat[i - 4] = b[1] / np.sqrt(s3 * xi[1, 1])
    
    ew = np.log(np.mean(np.exp(stat ** 2 / 2)))
    mw = np.sum(stat ** 2) / len(stat)
    qlr = np.max(stat ** 2)
    
    lame = _interpolate_lambda(ew, VAL_EW)
    lamm = _interpolate_lambda(mw, VAL_MW)
    lamq = _interpolate_lambda(qlr, VAL_QL)
    
    if np.isnan(lame) or np.isnan(lamm) or np.isnan(lamq):
        print("Warning: At least one statistic has an NA value.")
    
    return lame / (t_end - 1)


def median_unbiased_estimator_stage2(y: np.ndarray, x: np.ndarray, kappa_vec: np.ndarray) -> float:
    """Median unbiased estimation of lambda_z (Stock-Watson 1998)."""
    t_end = x.shape[0]
    stat = np.zeros(t_end - 2 * 4 + 1)
    w = np.diag(1 / (kappa_vec ** 2))
    
    for i in range(4, t_end - 3):
        xr = np.column_stack([x, np.concatenate([np.zeros(i), np.ones(t_end - i)])])
        xi = np.linalg.inv(xr.T @ w @ xr)
        b = np.linalg.solve(xr.T @ w @ xr, xr.T @ w @ y)
        s3 = np.sum(w @ (y - xr @ b) ** 2) / (np.sum(np.diag(w)) - xr.shape[1])
        stat[i - 4] = b[-1] / np.sqrt(s3 * xi[-1, -1])
    
    ew = np.log(np.mean(np.exp(stat ** 2 / 2)))
    mw = np.mean(stat ** 2)
    qlr = np.max(stat ** 2)
    
    lame = _interpolate_lambda(ew, VAL_EW)
    lamm = _interpolate_lambda(mw, VAL_MW)
    lamq = _interpolate_lambda(qlr, VAL_QL)
    
    if np.isnan(lame) or np.isnan(lamm) or np.isnan(lamq):
        print("Warning: At least one statistic has an NA value.")
    
    return lame / t_end


# =============================================================================
# STATE-SPACE PARAMETER BUILDERS
# =============================================================================

@dataclass
class StateSpaceMatrices:
    xi_00: np.ndarray
    P_00: np.ndarray
    F: np.ndarray
    Q: np.ndarray
    A: np.ndarray
    H: np.ndarray
    R: np.ndarray
    kappa_vec: np.ndarray
    cons: np.ndarray


def unpack_parameters_stage1(
    parameters: np.ndarray, y_data: np.ndarray, x_data: np.ndarray,
    xi_00: np.ndarray, P_00: np.ndarray, use_kappa: bool,
    kappa_inputs: Optional[pd.DataFrame], param_num: Dict[str, int],
) -> StateSpaceMatrices:
    n_state_vars = 3
    
    A = np.zeros((10, 2))
    A[0, 0] = parameters[param_num["a_1"] - 1]
    A[1, 0] = parameters[param_num["a_2"] - 1]
    A[0, 1] = parameters[param_num["b_3"] - 1]
    A[2, 1] = parameters[param_num["b_1"] - 1]
    A[3, 1] = parameters[param_num["b_2"] - 1]
    A[4, 1] = 1 - A[2, 1] - A[3, 1]
    A[5, 1] = parameters[param_num["b_4"] - 1]
    A[6, 1] = parameters[param_num["b_5"] - 1]
    A[7, 0] = parameters[param_num["phi"] - 1]
    A[8, 0] = -parameters[param_num["a_1"] - 1] * parameters[param_num["phi"] - 1]
    A[8, 1] = -parameters[param_num["b_3"] - 1] * parameters[param_num["phi"] - 1]
    A[9, 0] = -parameters[param_num["a_2"] - 1] * parameters[param_num["phi"] - 1]
    
    H = np.zeros((3, 2))
    H[0, 0] = 1
    H[1, 0] = -parameters[param_num["a_1"] - 1]
    H[2, 0] = -parameters[param_num["a_2"] - 1]
    H[1, 1] = -parameters[param_num["b_3"] - 1]
    
    R = np.diag([parameters[param_num["sigma_1"] - 1] ** 2, parameters[param_num["sigma_2"] - 1] ** 2])
    
    Q = np.zeros((3, 3))
    Q[0, 0] = parameters[param_num["sigma_4"] - 1] ** 2
    
    F = np.zeros((3, 3))
    F[0, 0] = 1
    F[1, 0] = 1
    F[2, 1] = 1
    
    kappa_vec = np.ones(y_data.shape[0])
    if use_kappa and kappa_inputs is not None:
        for idx, row in kappa_inputs.iterrows():
            T_start = int(row["T.start"]) - 1
            T_end = int(row["T.end"])
            theta_idx = int(row["theta.index"]) - 1
            kappa_vec[T_start:T_end] = parameters[theta_idx]
    
    cons = np.zeros((n_state_vars, 1))
    cons[0, 0] = parameters[param_num["g"] - 1]
    
    return StateSpaceMatrices(xi_00=xi_00, P_00=P_00, F=F, Q=Q, A=A, H=H, R=R, kappa_vec=kappa_vec, cons=cons)


def unpack_parameters_stage2(
    parameters: np.ndarray, y_data: np.ndarray, x_data: np.ndarray,
    lambda_g: float, xi_00: np.ndarray, P_00: np.ndarray, use_kappa: bool,
    kappa_inputs: Optional[pd.DataFrame], param_num: Dict[str, int],
) -> StateSpaceMatrices:
    n_state_vars = 6
    
    A = np.zeros((2, 13))
    A[0, 0] = parameters[param_num["a_1"] - 1]
    A[0, 1] = parameters[param_num["a_2"] - 1]
    A[0, 2] = parameters[param_num["a_3"] - 1] / 2
    A[0, 3] = parameters[param_num["a_3"] - 1] / 2
    A[0, 9] = parameters[param_num["a_4"] - 1]
    A[1, 0] = parameters[param_num["b_3"] - 1]
    A[1, 4] = parameters[param_num["b_1"] - 1]
    A[1, 5] = parameters[param_num["b_2"] - 1]
    A[1, 6] = 1 - parameters[param_num["b_1"] - 1] - parameters[param_num["b_2"] - 1]
    A[1, 7] = parameters[param_num["b_4"] - 1]
    A[1, 8] = parameters[param_num["b_5"] - 1]
    A[0, 10] = parameters[param_num["phi"] - 1]
    A[0, 11] = -parameters[param_num["a_1"] - 1] * parameters[param_num["phi"] - 1]
    A[0, 12] = -parameters[param_num["a_2"] - 1] * parameters[param_num["phi"] - 1]
    A[1, 11] = -parameters[param_num["b_3"] - 1] * parameters[param_num["phi"] - 1]
    A = A.T
    
    H = np.zeros((2, 6))
    H[0, 0] = 1
    H[0, 1] = -parameters[param_num["a_1"] - 1]
    H[0, 2] = -parameters[param_num["a_2"] - 1]
    H[0, 4] = parameters[param_num["a_5"] - 1] / 2
    H[0, 5] = parameters[param_num["a_5"] - 1] / 2
    H[1, 1] = -parameters[param_num["b_3"] - 1]
    H = H.T
    
    R = np.diag([parameters[param_num["sigma_1"] - 1] ** 2, parameters[param_num["sigma_2"] - 1] ** 2])
    
    Q = np.zeros((6, 6))
    Q[0, 0] = parameters[param_num["sigma_4"] - 1] ** 2
    Q[3, 3] = (lambda_g * parameters[param_num["sigma_4"] - 1]) ** 2
    
    F = np.zeros((6, 6))
    F[0, 0] = 1
    F[0, 3] = 1
    F[1, 0] = 1
    F[2, 1] = 1
    F[3, 3] = 1
    F[4, 3] = 1
    F[5, 4] = 1
    
    kappa_vec = np.ones(y_data.shape[0])
    if use_kappa and kappa_inputs is not None:
        for idx, row in kappa_inputs.iterrows():
            T_start = int(row["T.start"]) - 1
            T_end = int(row["T.end"])
            theta_idx = int(row["theta.index"]) - 1
            kappa_vec[T_start:T_end] = parameters[theta_idx]
    
    cons = np.zeros((n_state_vars, 1))
    return StateSpaceMatrices(xi_00=xi_00, P_00=P_00, F=F, Q=Q, A=A, H=H, R=R, kappa_vec=kappa_vec, cons=cons)


def unpack_parameters_stage3(
    parameters: np.ndarray, y_data: np.ndarray, x_data: np.ndarray,
    lambda_g: float, lambda_z: float, xi_00: np.ndarray, P_00: np.ndarray,
    use_kappa: bool, kappa_inputs: Optional[pd.DataFrame], param_num: Dict[str, int],
) -> StateSpaceMatrices:
    n_state_vars = 9
    
    A = np.zeros((2, 12))
    A[0, 0] = parameters[param_num["a_1"] - 1]
    A[0, 1] = parameters[param_num["a_2"] - 1]
    A[0, 2] = parameters[param_num["a_3"] - 1] / 2
    A[0, 3] = parameters[param_num["a_3"] - 1] / 2
    A[1, 0] = parameters[param_num["b_3"] - 1]
    A[1, 4] = parameters[param_num["b_1"] - 1]
    A[1, 5] = parameters[param_num["b_2"] - 1]
    A[1, 6] = 1 - parameters[param_num["b_1"] - 1] - parameters[param_num["b_2"] - 1]
    A[1, 7] = parameters[param_num["b_4"] - 1]
    A[1, 8] = parameters[param_num["b_5"] - 1]
    A[0, 9] = parameters[param_num["phi"] - 1]
    A[0, 10] = -parameters[param_num["a_1"] - 1] * parameters[param_num["phi"] - 1]
    A[0, 11] = -parameters[param_num["a_2"] - 1] * parameters[param_num["phi"] - 1]
    A[1, 10] = -parameters[param_num["b_3"] - 1] * parameters[param_num["phi"] - 1]
    A = A.T
    
    H = np.zeros((2, 9))
    H[0, 0] = 1
    H[0, 1] = -parameters[param_num["a_1"] - 1]
    H[0, 2] = -parameters[param_num["a_2"] - 1]
    H[0, 4] = -parameters[param_num["c"] - 1] * parameters[param_num["a_3"] - 1] * 2
    H[0, 5] = -parameters[param_num["c"] - 1] * parameters[param_num["a_3"] - 1] * 2
    H[0, 7] = -parameters[param_num["a_3"] - 1] / 2
    H[0, 8] = -parameters[param_num["a_3"] - 1] / 2
    H[1, 1] = -parameters[param_num["b_3"] - 1]
    H = H.T
    
    R = np.diag([parameters[param_num["sigma_1"] - 1] ** 2, parameters[param_num["sigma_2"] - 1] ** 2])
    
    Q = np.zeros((9, 9))
    Q[0, 0] = parameters[param_num["sigma_4"] - 1] ** 2
    Q[3, 3] = (lambda_g * parameters[param_num["sigma_4"] - 1]) ** 2
    Q[6, 6] = (lambda_z * parameters[param_num["sigma_1"] - 1] / parameters[param_num["a_3"] - 1]) ** 2
    
    F = np.zeros((9, 9))
    F[0, 0] = 1
    F[0, 3] = 1
    F[1, 0] = 1
    F[2, 1] = 1
    F[3, 3] = 1
    F[4, 3] = 1
    F[5, 4] = 1
    F[6, 6] = 1
    F[7, 6] = 1
    F[8, 7] = 1
    
    kappa_vec = np.ones(y_data.shape[0])
    if use_kappa and kappa_inputs is not None:
        for idx, row in kappa_inputs.iterrows():
            T_start = int(row["T.start"]) - 1
            T_end = int(row["T.end"])
            theta_idx = int(row["theta.index"]) - 1
            kappa_vec[T_start:T_end] = parameters[theta_idx]
    
    cons = np.zeros((n_state_vars, 1))
    return StateSpaceMatrices(xi_00=xi_00, P_00=P_00, F=F, Q=Q, A=A, H=H, R=R, kappa_vec=kappa_vec, cons=cons)


# =============================================================================
# STAGE RESULT DATACLASSES
# =============================================================================

@dataclass
class Stage1Result:
    theta: np.ndarray
    log_likelihood: float
    states: KalmanStates
    matrices: StateSpaceMatrices
    xi_00: np.ndarray
    P_00: np.ndarray
    potential_filtered: np.ndarray
    output_gap_filtered: np.ndarray
    potential_smoothed: np.ndarray
    output_gap_smoothed: np.ndarray


@dataclass
class Stage2Result:
    theta: np.ndarray
    log_likelihood: float
    states: KalmanStates
    matrices: StateSpaceMatrices
    xi_00: np.ndarray
    P_00: np.ndarray
    y: np.ndarray
    x: np.ndarray
    kappa_vec: np.ndarray
    trend_filtered: np.ndarray
    potential_filtered: np.ndarray
    output_gap_filtered: np.ndarray
    trend_smoothed: np.ndarray
    potential_smoothed: np.ndarray
    output_gap_smoothed: np.ndarray


@dataclass
class Stage3Result:
    theta: np.ndarray
    log_likelihood: float
    lambda_g: float
    lambda_z: float
    states: KalmanStates
    matrices: StateSpaceMatrices
    xi_00: np.ndarray
    P_00: np.ndarray
    rstar_filtered: np.ndarray
    trend_filtered: np.ndarray
    z_filtered: np.ndarray
    potential_filtered: np.ndarray
    output_gap_filtered: np.ndarray
    rstar_smoothed: np.ndarray
    trend_smoothed: np.ndarray
    z_smoothed: np.ndarray
    potential_smoothed: np.ndarray
    output_gap_smoothed: np.ndarray


# =============================================================================
# LIKELIHOOD AND STATE WRAPPERS
# =============================================================================

def _log_likelihood_wrapper(
    parameters: np.ndarray, y_data: np.ndarray, x_data: np.ndarray,
    stage: int, lambda_g: Optional[float], lambda_z: Optional[float],
    xi_00: np.ndarray, P_00: np.ndarray, use_kappa: bool,
    kappa_inputs: Optional[pd.DataFrame], param_num: Dict[str, int],
) -> float:
    if stage == 1:
        matrices = unpack_parameters_stage1(parameters, y_data, x_data, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    elif stage == 2:
        matrices = unpack_parameters_stage2(parameters, y_data, x_data, lambda_g, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    elif stage == 3:
        matrices = unpack_parameters_stage3(parameters, y_data, x_data, lambda_g, lambda_z, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    _, ll_cum, _ = kalman_log_likelihood(
        matrices.xi_00, matrices.P_00, matrices.F, matrices.Q,
        matrices.A, matrices.H, matrices.R, matrices.kappa_vec, matrices.cons, y_data, x_data
    )
    return ll_cum


def _kalman_states_wrapper(
    parameters: np.ndarray, y_data: np.ndarray, x_data: np.ndarray,
    stage: int, lambda_g: Optional[float], lambda_z: Optional[float],
    xi_00: np.ndarray, P_00: np.ndarray, use_kappa: bool,
    kappa_inputs: Optional[pd.DataFrame], param_num: Dict[str, int],
) -> Tuple[KalmanStates, StateSpaceMatrices]:
    if stage == 1:
        matrices = unpack_parameters_stage1(parameters, y_data, x_data, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    elif stage == 2:
        matrices = unpack_parameters_stage2(parameters, y_data, x_data, lambda_g, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    elif stage == 3:
        matrices = unpack_parameters_stage3(parameters, y_data, x_data, lambda_g, lambda_z, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    states = kalman_states(
        matrices.xi_00, matrices.P_00, matrices.F, matrices.Q,
        matrices.A, matrices.H, matrices.R, matrices.kappa_vec, matrices.cons, y_data, x_data
    )
    return states, matrices


def calculate_covariance(
    initial_parameters: np.ndarray, theta_lb: np.ndarray, theta_ub: np.ndarray,
    y_data: np.ndarray, x_data: np.ndarray, stage: int,
    lambda_g: Optional[float], lambda_z: Optional[float], xi_00: np.ndarray,
    use_kappa: bool, kappa_inputs: Optional[pd.DataFrame], param_num: Dict[str, int],
) -> np.ndarray:
    n_state_vars = len(xi_00)
    P_00 = np.eye(n_state_vars) * 0.2
    
    def neg_ll(theta):
        return -_log_likelihood_wrapper(theta, y_data, x_data, stage, lambda_g, lambda_z, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    
    bounds = list(zip(theta_lb, theta_ub))
    result = minimize(neg_ll, initial_parameters, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000, "ftol": 1e-8})
    
    if not result.success:
        print(f"Warning: calculate_covariance: {result.message}")
    
    theta = result.x
    states, _ = _kalman_states_wrapper(theta, y_data, x_data, stage, lambda_g, lambda_z, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    return states.filtered.P_ttm1[:n_state_vars, :]


# =============================================================================
# STAGE 1 ESTIMATION
# =============================================================================

def rstar_stage1(
    log_output: np.ndarray, inflation: np.ndarray, relative_oil_price_inflation: np.ndarray,
    relative_import_price_inflation: np.ndarray, covid_dummy: np.ndarray,
    sample_end: Tuple[int, int], b_y_constraint: Optional[float] = None,
    xi_00: Optional[np.ndarray] = None, P_00: Optional[np.ndarray] = None,
    use_kappa: bool = False, kappa_inputs: Optional[pd.DataFrame] = None, fix_phi: Optional[float] = None,
) -> Stage1Result:
    stage = 1
    t_end = len(log_output) - 8
    
    x_og = np.column_stack([
        np.ones(t_end + 4), np.arange(1, t_end + 5),
        np.concatenate([np.zeros(56), np.arange(1, t_end + 4 - 56 + 1)]) if t_end + 4 > 56 else np.zeros(t_end + 4),
        np.concatenate([np.zeros(142), np.arange(1, t_end + 4 - 142 + 1)]) if t_end + 4 > 142 else np.zeros(t_end + 4),
    ])
    y_og = log_output[4:t_end + 8]
    output_gap = (y_og - x_og @ np.linalg.solve(x_og.T @ x_og, x_og.T @ y_og)) * 100
    
    if xi_00 is None:
        print("Stage 1: xi.00 from HP trend")
        _, g_pot = hpfilter(y_og, lamb=36000)
        xi_00 = np.array([100 * g_pot[3], 100 * g_pot[2], 100 * g_pot[1]])
    
    y_is = output_gap[4:t_end + 4]
    y_is_l1 = output_gap[3:t_end + 3]
    y_is_l2 = output_gap[2:t_end + 2]
    d = covid_dummy[8:t_end + 8]
    d_l1 = covid_dummy[7:t_end + 7]
    d_l2 = covid_dummy[6:t_end + 6]
    
    if sample_end[0] >= 2020 and fix_phi is None:
        def is_residuals(params):
            phi, a1, a2 = params
            return y_is - (phi * d + a1 * (y_is_l1 - phi * d_l1) + a2 * (y_is_l2 - phi * d_l2))
        result = least_squares(is_residuals, [0, 0, 0])
        b_is = {"phi": result.x[0], "a_1": result.x[1], "a_2": result.x[2]}
    else:
        def is_residuals(params):
            a1, a2 = params
            return y_is - (a1 * y_is_l1 + a2 * y_is_l2)
        result = least_squares(is_residuals, [0, 0])
        b_is = {"a_1": result.x[0], "a_2": result.x[1], "phi": 0.0}
    
    if fix_phi is not None:
        b_is["phi"] = fix_phi
    
    r_is = is_residuals([b_is.get("phi", 0), b_is["a_1"], b_is["a_2"]] if "phi" in b_is and b_is["phi"] != 0 else [b_is["a_1"], b_is["a_2"]])
    s_is = np.sqrt(np.sum(r_is ** 2) / (len(r_is) - len(b_is)))
    
    y_ph = inflation[8:t_end + 8]
    x_ph = np.column_stack([
        inflation[7:t_end + 7],
        (inflation[6:t_end + 6] + inflation[5:t_end + 5] + inflation[4:t_end + 4]) / 3,
        (inflation[3:t_end + 3] + inflation[2:t_end + 2] + inflation[1:t_end + 1] + inflation[0:t_end]) / 4,
        y_is_l1 - b_is["phi"] * d_l1,
        relative_oil_price_inflation[7:t_end + 7],
        relative_import_price_inflation[8:t_end + 8],
    ])
    b_ph = np.linalg.solve(x_ph.T @ x_ph, x_ph.T @ y_ph)
    r_ph = y_ph - x_ph @ b_ph
    s_ph = np.sqrt(np.sum(r_ph ** 2) / (len(r_ph) - len(b_ph)))
    
    initial_parameters = np.array([b_is["a_1"], b_is["a_2"], b_ph[0], b_ph[1], b_ph[3], b_ph[4], b_ph[5], 0.85, s_is, s_ph, 0.5, b_is["phi"]])
    param_num = {"a_1": 1, "a_2": 2, "b_1": 3, "b_2": 4, "b_3": 5, "b_4": 6, "b_5": 7, "g": 8, "sigma_1": 9, "sigma_2": 10, "sigma_4": 11, "phi": 12}
    n_params = len(initial_parameters)
    
    y_data = np.column_stack([100 * log_output[8:t_end + 8], inflation[8:t_end + 8]])
    x_data = np.column_stack([
        100 * log_output[7:t_end + 7], 100 * log_output[6:t_end + 6],
        inflation[7:t_end + 7],
        (inflation[6:t_end + 6] + inflation[5:t_end + 5] + inflation[4:t_end + 4]) / 3,
        (inflation[3:t_end + 3] + inflation[2:t_end + 2] + inflation[1:t_end + 1] + inflation[0:t_end]) / 4,
        relative_oil_price_inflation[7:t_end + 7], relative_import_price_inflation[8:t_end + 8],
        covid_dummy[8:t_end + 8], covid_dummy[7:t_end + 7], covid_dummy[6:t_end + 6],
    ])
    
    theta_lb = np.full(n_params, -np.inf)
    theta_ub = np.full(n_params, np.inf)
    if b_y_constraint is not None:
        if initial_parameters[param_num["b_3"] - 1] < b_y_constraint:
            initial_parameters[param_num["b_3"] - 1] = b_y_constraint
        theta_lb[param_num["b_3"] - 1] = b_y_constraint
    if fix_phi is not None:
        theta_lb[param_num["phi"] - 1] = fix_phi
        theta_ub[param_num["phi"] - 1] = fix_phi
    
    if use_kappa and kappa_inputs is not None:
        for k, row in kappa_inputs.iterrows():
            theta_ind = n_params + k
            kappa_inputs.loc[k, "theta.index"] = theta_ind + 1
            param_num[row["name"]] = theta_ind + 1
            initial_parameters = np.append(initial_parameters, row["init"])
            theta_lb = np.append(theta_lb, row["lower.bound"])
            theta_ub = np.append(theta_ub, row["upper.bound"])
    
    if P_00 is None:
        P_00 = calculate_covariance(initial_parameters, theta_lb, theta_ub, y_data, x_data, stage, None, None, xi_00, use_kappa, kappa_inputs, param_num)
    
    def neg_ll(theta):
        return -_log_likelihood_wrapper(theta, y_data, x_data, stage, None, None, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    
    bounds = list(zip(theta_lb, theta_ub))
    result = minimize(neg_ll, initial_parameters, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000, "ftol": 1e-8})
    print(f"Stage 1: {result.message}")
    
    theta = result.x
    log_likelihood = -result.fun
    states, matrices = _kalman_states_wrapper(theta, y_data, x_data, stage, None, None, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    
    potential_filtered = states.filtered.xi_tt[:, 0] / 100
    output_gap_filtered = y_data[:, 0] - potential_filtered * 100 - theta[param_num["phi"] - 1] * covid_dummy[8:t_end + 8]
    potential_smoothed = states.smoothed.xi_tT[:, 0] / 100
    output_gap_smoothed = y_data[:, 0] - potential_smoothed * 100 - theta[param_num["phi"] - 1] * covid_dummy[8:t_end + 8]
    
    return Stage1Result(theta=theta, log_likelihood=log_likelihood, states=states, matrices=matrices, xi_00=xi_00, P_00=P_00,
                        potential_filtered=potential_filtered, output_gap_filtered=output_gap_filtered,
                        potential_smoothed=potential_smoothed, output_gap_smoothed=output_gap_smoothed)


# =============================================================================
# STAGE 2 ESTIMATION
# =============================================================================

def rstar_stage2(
    log_output: np.ndarray, inflation: np.ndarray, relative_oil_price_inflation: np.ndarray,
    relative_import_price_inflation: np.ndarray, real_interest_rate: np.ndarray, covid_dummy: np.ndarray,
    lambda_g: float, sample_end: Tuple[int, int], a_r_constraint: Optional[float] = None,
    b_y_constraint: Optional[float] = None, xi_00: Optional[np.ndarray] = None,
    P_00: Optional[np.ndarray] = None, use_kappa: bool = False,
    kappa_inputs: Optional[pd.DataFrame] = None, fix_phi: Optional[float] = None,
) -> Stage2Result:
    stage = 2
    t_end = len(log_output) - 8
    
    x_og = np.column_stack([
        np.ones(t_end + 4), np.arange(1, t_end + 5),
        np.concatenate([np.zeros(56), np.arange(1, t_end + 4 - 56 + 1)]) if t_end + 4 > 56 else np.zeros(t_end + 4),
        np.concatenate([np.zeros(142), np.arange(1, t_end + 4 - 142 + 1)]) if t_end + 4 > 142 else np.zeros(t_end + 4),
    ])
    y_og = log_output[4:t_end + 8]
    output_gap = (y_og - x_og @ np.linalg.solve(x_og.T @ x_og, x_og.T @ y_og)) * 100
    
    if xi_00 is None:
        _, g_pot = hpfilter(y_og, lamb=36000)
        g_pot_diff = np.diff(g_pot)
        xi_00 = np.array([100 * g_pot[3], 100 * g_pot[2], 100 * g_pot[1], 100 * g_pot_diff[2], 100 * g_pot_diff[1], 100 * g_pot_diff[0]])
    
    y_is = output_gap[4:t_end + 4]
    y_is_l1 = output_gap[3:t_end + 3]
    y_is_l2 = output_gap[2:t_end + 2]
    d = covid_dummy[8:t_end + 8]
    d_l1 = covid_dummy[7:t_end + 7]
    d_l2 = covid_dummy[6:t_end + 6]
    ir_is = (real_interest_rate[7:t_end + 7] + real_interest_rate[6:t_end + 6]) / 2
    
    if sample_end[0] >= 2020 and fix_phi is None:
        def is_residuals(params):
            phi, a1, a2, ar, a0 = params
            return y_is - (phi * d + a1 * (y_is_l1 - phi * d_l1) + a2 * (y_is_l2 - phi * d_l2) + ar * ir_is + a0)
        result = least_squares(is_residuals, [0, 0, 0, 0, 0])
        b_is = {"phi": result.x[0], "a_1": result.x[1], "a_2": result.x[2], "a_r": result.x[3], "a_0": result.x[4]}
    else:
        def is_residuals(params):
            a1, a2, ar, a0 = params
            return y_is - (a1 * y_is_l1 + a2 * y_is_l2 + ar * ir_is + a0)
        result = least_squares(is_residuals, [0, 0, 0, 0])
        b_is = {"a_1": result.x[0], "a_2": result.x[1], "a_r": result.x[2], "a_0": result.x[3], "phi": 0.0}
    
    if fix_phi is not None:
        b_is["phi"] = fix_phi
    
    r_is = y_is - (b_is["phi"] * d + b_is["a_1"] * (y_is_l1 - b_is["phi"] * d_l1) + b_is["a_2"] * (y_is_l2 - b_is["phi"] * d_l2) + b_is["a_r"] * ir_is + b_is["a_0"])
    s_is = np.sqrt(np.sum(r_is ** 2) / (len(r_is) - len(b_is)))
    
    y_ph = inflation[8:t_end + 8]
    x_ph = np.column_stack([
        inflation[7:t_end + 7],
        (inflation[6:t_end + 6] + inflation[5:t_end + 5] + inflation[4:t_end + 4]) / 3,
        (inflation[3:t_end + 3] + inflation[2:t_end + 2] + inflation[1:t_end + 1] + inflation[0:t_end]) / 4,
        y_is_l1 - b_is["phi"] * d_l1,
        relative_oil_price_inflation[7:t_end + 7], relative_import_price_inflation[8:t_end + 8],
    ])
    b_ph = np.linalg.solve(x_ph.T @ x_ph, x_ph.T @ y_ph)
    r_ph = y_ph - x_ph @ b_ph
    s_ph = np.sqrt(np.sum(r_ph ** 2) / (len(r_ph) - len(b_ph)))
    
    initial_parameters = np.array([b_is["a_1"], b_is["a_2"], b_is["a_r"], b_is["a_0"], -b_is["a_r"], b_ph[0], b_ph[1], b_ph[3], b_ph[4], b_ph[5], s_is, s_ph, 0.5, b_is["phi"]])
    param_num = {"a_1": 1, "a_2": 2, "a_3": 3, "a_4": 4, "a_5": 5, "b_1": 6, "b_2": 7, "b_3": 8, "b_4": 9, "b_5": 10, "sigma_1": 11, "sigma_2": 12, "sigma_4": 13, "phi": 14}
    n_params = len(initial_parameters)
    
    y_data = np.column_stack([100 * log_output[8:t_end + 8], inflation[8:t_end + 8]])
    x_data = np.column_stack([
        100 * log_output[7:t_end + 7], 100 * log_output[6:t_end + 6],
        real_interest_rate[7:t_end + 7], real_interest_rate[6:t_end + 6],
        inflation[7:t_end + 7],
        (inflation[6:t_end + 6] + inflation[5:t_end + 5] + inflation[4:t_end + 4]) / 3,
        (inflation[3:t_end + 3] + inflation[2:t_end + 2] + inflation[1:t_end + 1] + inflation[0:t_end]) / 4,
        relative_oil_price_inflation[7:t_end + 7], relative_import_price_inflation[8:t_end + 8],
        np.ones(t_end), covid_dummy[8:t_end + 8], covid_dummy[7:t_end + 7], covid_dummy[6:t_end + 6],
    ])
    
    theta_lb = np.full(n_params, -np.inf)
    theta_ub = np.full(n_params, np.inf)
    if b_y_constraint is not None:
        if initial_parameters[param_num["b_3"] - 1] < b_y_constraint:
            initial_parameters[param_num["b_3"] - 1] = b_y_constraint
        theta_lb[param_num["b_3"] - 1] = b_y_constraint
    if a_r_constraint is not None:
        if initial_parameters[param_num["a_3"] - 1] > a_r_constraint:
            initial_parameters[param_num["a_3"] - 1] = a_r_constraint
        theta_ub[param_num["a_3"] - 1] = a_r_constraint
    if fix_phi is not None:
        theta_lb[param_num["phi"] - 1] = fix_phi
        theta_ub[param_num["phi"] - 1] = fix_phi
    
    if use_kappa and kappa_inputs is not None:
        for k, row in kappa_inputs.iterrows():
            theta_ind = n_params + k
            kappa_inputs.loc[k, "theta.index"] = theta_ind + 1
            param_num[row["name"]] = theta_ind + 1
            initial_parameters = np.append(initial_parameters, row["init"])
            theta_lb = np.append(theta_lb, row["lower.bound"])
            theta_ub = np.append(theta_ub, row["upper.bound"])
    
    if P_00 is None:
        P_00 = calculate_covariance(initial_parameters, theta_lb, theta_ub, y_data, x_data, stage, lambda_g, None, xi_00, use_kappa, kappa_inputs, param_num)
    
    def neg_ll(theta):
        return -_log_likelihood_wrapper(theta, y_data, x_data, stage, lambda_g, None, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    
    bounds = list(zip(theta_lb, theta_ub))
    result = minimize(neg_ll, initial_parameters, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000, "ftol": 1e-8})
    print(f"Stage 2: {result.message}")
    
    theta = result.x
    log_likelihood = -result.fun
    states, matrices = _kalman_states_wrapper(theta, y_data, x_data, stage, lambda_g, None, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    
    trend_smoothed = states.smoothed.xi_tT[:, 3] * 4
    potential_smoothed_full = np.concatenate([states.smoothed.xi_tT[0, 2:0:-1], states.smoothed.xi_tT[:, 0]])
    output_gap_smoothed = 100 * log_output[6:t_end + 8] - potential_smoothed_full - theta[param_num["phi"] - 1] * covid_dummy[6:t_end + 8]
    
    y_lambda = output_gap_smoothed[2:]
    x_lambda = np.column_stack([output_gap_smoothed[1:-1], output_gap_smoothed[:-2], (x_data[:, 2] + x_data[:, 3]) / 2, trend_smoothed, np.ones(t_end)])
    
    trend_filtered = states.filtered.xi_tt[:, 3] * 4
    potential_filtered = states.filtered.xi_tt[:, 0] / 100
    output_gap_filtered = y_data[:, 0] - potential_filtered * 100 - theta[param_num["phi"] - 1] * covid_dummy[8:t_end + 8]
    
    return Stage2Result(theta=theta, log_likelihood=log_likelihood, states=states, matrices=matrices, xi_00=xi_00, P_00=P_00,
                        y=y_lambda, x=x_lambda, kappa_vec=matrices.kappa_vec,
                        trend_filtered=trend_filtered, potential_filtered=potential_filtered, output_gap_filtered=output_gap_filtered,
                        trend_smoothed=trend_smoothed, potential_smoothed=potential_smoothed_full[2:], output_gap_smoothed=output_gap_smoothed[2:])


# =============================================================================
# STAGE 3 ESTIMATION
# =============================================================================

def rstar_stage3(
    log_output: np.ndarray, inflation: np.ndarray, relative_oil_price_inflation: np.ndarray,
    relative_import_price_inflation: np.ndarray, real_interest_rate: np.ndarray, covid_dummy: np.ndarray,
    lambda_g: float, lambda_z: float, sample_end: Tuple[int, int], a_r_constraint: Optional[float] = None,
    b_y_constraint: Optional[float] = None, xi_00: Optional[np.ndarray] = None,
    P_00: Optional[np.ndarray] = None, use_kappa: bool = False,
    kappa_inputs: Optional[pd.DataFrame] = None, fix_phi: Optional[float] = None,
) -> Stage3Result:
    stage = 3
    t_end = len(log_output) - 8
    
    x_og = np.column_stack([
        np.ones(t_end + 4), np.arange(1, t_end + 5),
        np.concatenate([np.zeros(56), np.arange(1, t_end + 4 - 56 + 1)]) if t_end + 4 > 56 else np.zeros(t_end + 4),
        np.concatenate([np.zeros(142), np.arange(1, t_end + 4 - 142 + 1)]) if t_end + 4 > 142 else np.zeros(t_end + 4),
    ])
    y_og = log_output[4:t_end + 8]
    output_gap = (y_og - x_og @ np.linalg.solve(x_og.T @ x_og, x_og.T @ y_og)) * 100
    
    y_is = output_gap[4:t_end + 4]
    y_is_l1 = output_gap[3:t_end + 3]
    y_is_l2 = output_gap[2:t_end + 2]
    d = covid_dummy[8:t_end + 8]
    d_l1 = covid_dummy[7:t_end + 7]
    d_l2 = covid_dummy[6:t_end + 6]
    ir_is = (real_interest_rate[7:t_end + 7] + real_interest_rate[6:t_end + 6]) / 2
    
    if sample_end[0] >= 2020 and fix_phi is None:
        def is_residuals(params):
            phi, a1, a2, ar, a0 = params
            return y_is - (phi * d + a1 * (y_is_l1 - phi * d_l1) + a2 * (y_is_l2 - phi * d_l2) + ar * ir_is + a0)
        result = least_squares(is_residuals, [0, 0, 0, 0, 0])
        b_is = {"phi": result.x[0], "a_1": result.x[1], "a_2": result.x[2], "a_r": result.x[3], "a_0": result.x[4]}
    else:
        def is_residuals(params):
            a1, a2, ar, a0 = params
            return y_is - (a1 * y_is_l1 + a2 * y_is_l2 + ar * ir_is + a0)
        result = least_squares(is_residuals, [0, 0, 0, 0])
        b_is = {"a_1": result.x[0], "a_2": result.x[1], "a_r": result.x[2], "a_0": result.x[3], "phi": 0.0}
    
    if fix_phi is not None:
        b_is["phi"] = fix_phi
    
    r_is = y_is - (b_is["phi"] * d + b_is["a_1"] * (y_is_l1 - b_is["phi"] * d_l1) + b_is["a_2"] * (y_is_l2 - b_is["phi"] * d_l2) + b_is["a_r"] * ir_is + b_is["a_0"])
    s_is = np.sqrt(np.sum(r_is ** 2) / (len(r_is) - len(b_is)))
    
    y_ph = inflation[8:t_end + 8]
    x_ph = np.column_stack([
        inflation[7:t_end + 7],
        (inflation[6:t_end + 6] + inflation[5:t_end + 5] + inflation[4:t_end + 4]) / 3,
        (inflation[3:t_end + 3] + inflation[2:t_end + 2] + inflation[1:t_end + 1] + inflation[0:t_end]) / 4,
        y_is_l1 - b_is["phi"] * d_l1,
        relative_oil_price_inflation[7:t_end + 7], relative_import_price_inflation[8:t_end + 8],
    ])
    b_ph = np.linalg.solve(x_ph.T @ x_ph, x_ph.T @ y_ph)
    r_ph = y_ph - x_ph @ b_ph
    s_ph = np.sqrt(np.sum(r_ph ** 2) / (len(r_ph) - len(b_ph)))
    
    initial_parameters = np.array([b_is["a_1"], b_is["a_2"], b_is["a_r"], b_ph[0], b_ph[1], b_ph[3], b_ph[4], b_ph[5], 1.0, s_is, s_ph, 0.7, b_is["phi"]])
    param_num = {"a_1": 1, "a_2": 2, "a_3": 3, "b_1": 4, "b_2": 5, "b_3": 6, "b_4": 7, "b_5": 8, "c": 9, "sigma_1": 10, "sigma_2": 11, "sigma_4": 12, "phi": 13}
    n_params = len(initial_parameters)
    
    if xi_00 is None:
        _, g_pot = hpfilter(y_og, lamb=36000)
        g_pot_diff = np.diff(g_pot)
        xi_00 = np.array([100 * g_pot[3], 100 * g_pot[2], 100 * g_pot[1], 100 * g_pot_diff[2], 100 * g_pot_diff[1], 100 * g_pot_diff[0], 0, 0, 0])
    
    y_data = np.column_stack([100 * log_output[8:t_end + 8], inflation[8:t_end + 8]])
    x_data = np.column_stack([
        100 * log_output[7:t_end + 7], 100 * log_output[6:t_end + 6],
        real_interest_rate[7:t_end + 7], real_interest_rate[6:t_end + 6],
        inflation[7:t_end + 7],
        (inflation[6:t_end + 6] + inflation[5:t_end + 5] + inflation[4:t_end + 4]) / 3,
        (inflation[3:t_end + 3] + inflation[2:t_end + 2] + inflation[1:t_end + 1] + inflation[0:t_end]) / 4,
        relative_oil_price_inflation[7:t_end + 7], relative_import_price_inflation[8:t_end + 8],
        covid_dummy[8:t_end + 8], covid_dummy[7:t_end + 7], covid_dummy[6:t_end + 6],
    ])
    
    theta_lb = np.full(n_params, -np.inf)
    theta_ub = np.full(n_params, np.inf)
    if b_y_constraint is not None:
        if initial_parameters[param_num["b_3"] - 1] < b_y_constraint:
            initial_parameters[param_num["b_3"] - 1] = b_y_constraint
        theta_lb[param_num["b_3"] - 1] = b_y_constraint
    if a_r_constraint is not None:
        if initial_parameters[param_num["a_3"] - 1] > a_r_constraint:
            initial_parameters[param_num["a_3"] - 1] = a_r_constraint
        theta_ub[param_num["a_3"] - 1] = a_r_constraint
    if fix_phi is not None:
        theta_lb[param_num["phi"] - 1] = fix_phi
        theta_ub[param_num["phi"] - 1] = fix_phi
    
    if use_kappa and kappa_inputs is not None:
        for k, row in kappa_inputs.iterrows():
            theta_ind = n_params + k
            kappa_inputs.loc[k, "theta.index"] = theta_ind + 1
            param_num[row["name"]] = theta_ind + 1
            initial_parameters = np.append(initial_parameters, row["init"])
            theta_lb = np.append(theta_lb, row["lower.bound"])
            theta_ub = np.append(theta_ub, row["upper.bound"])
    
    if P_00 is None:
        P_00 = calculate_covariance(initial_parameters, theta_lb, theta_ub, y_data, x_data, stage, lambda_g, lambda_z, xi_00, use_kappa, kappa_inputs, param_num)
    
    def neg_ll(theta):
        return -_log_likelihood_wrapper(theta, y_data, x_data, stage, lambda_g, lambda_z, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    
    bounds = list(zip(theta_lb, theta_ub))
    result = minimize(neg_ll, initial_parameters, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000, "ftol": 1e-8})
    print(f"Stage 3: {result.message}")
    
    theta = result.x
    log_likelihood = -result.fun
    states, matrices = _kalman_states_wrapper(theta, y_data, x_data, stage, lambda_g, lambda_z, xi_00, P_00, use_kappa, kappa_inputs, param_num)
    
    trend_filtered = states.filtered.xi_tt[:, 3] * 4
    z_filtered = states.filtered.xi_tt[:, 6]
    rstar_filtered = trend_filtered * theta[param_num["c"] - 1] + z_filtered
    potential_filtered = states.filtered.xi_tt[:, 0] / 100
    output_gap_filtered = y_data[:, 0] - potential_filtered * 100 - theta[param_num["phi"] - 1] * covid_dummy[8:t_end + 8]
    
    trend_smoothed = states.smoothed.xi_tT[:, 3] * 4
    z_smoothed = states.smoothed.xi_tT[:, 6]
    rstar_smoothed = trend_smoothed * theta[param_num["c"] - 1] + z_smoothed
    potential_smoothed = states.smoothed.xi_tT[:, 0] / 100
    output_gap_smoothed = y_data[:, 0] - potential_smoothed * 100 - theta[param_num["phi"] - 1] * covid_dummy[8:t_end + 8]
    
    return Stage3Result(theta=theta, log_likelihood=log_likelihood, lambda_g=lambda_g, lambda_z=lambda_z,
                        states=states, matrices=matrices, xi_00=xi_00, P_00=P_00,
                        rstar_filtered=rstar_filtered, trend_filtered=trend_filtered, z_filtered=z_filtered,
                        potential_filtered=potential_filtered, output_gap_filtered=output_gap_filtered,
                        rstar_smoothed=rstar_smoothed, trend_smoothed=trend_smoothed, z_smoothed=z_smoothed,
                        potential_smoothed=potential_smoothed, output_gap_smoothed=output_gap_smoothed)


# =============================================================================
# MAIN RUN FUNCTION
# =============================================================================

def run_estimation(
    excel_path: str | Path,
    output_dir: str | Path = "outputs",
    sample_start: Tuple[int, int] = (1961, 1),
    sample_end: Tuple[int, int] = (2025, 2),
    use_kappa: bool = True,
    fix_phi: float | None = None,
) -> Dict[str, float]:
    """Run the full LW estimation."""
    excel_path = Path(excel_path)
    outputs_path = Path(output_dir)
    data_path = outputs_path / "data"
    figures_path = outputs_path / "figures"
    data_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    a_r_constraint = -0.0025
    b_y_constraint = 0.025
    
    data = pd.read_excel(excel_path, sheet_name="input data")
    data["Date"] = pd.to_datetime(data["Date"])
    
    est_data_start_year = sample_start[0] - 2 if sample_start[1] == 1 else sample_start[0] - 1
    est_data_start_quarter = (sample_start[1] - 8 - 1) % 4 + 1
    if sample_start[1] <= 8:
        est_data_start_year = sample_start[0] - (9 - sample_start[1]) // 4 - 1
        est_data_start_quarter = (sample_start[1] - 8 - 1) % 4 + 1
    
    start_date = pd.Timestamp(year=est_data_start_year, month=est_data_start_quarter * 3, day=1)
    end_date = pd.Timestamp(year=sample_end[0], month=sample_end[1] * 3, day=1)
    mask = (data["Date"] >= start_date) & (data["Date"] <= end_date)
    data = data[mask].copy().reset_index(drop=True)
    
    log_output = data["gdp.log"].values
    inflation = data["inflation"].values
    relative_oil_price_inflation = data["oil.price.inflation"].values - inflation
    relative_import_price_inflation = data["import.price.inflation"].values - inflation
    nominal_interest_rate = data["interest"].values
    inflation_expectations = data["inflation.expectations"].values
    covid_dummy = data["covid.ind"].values
    real_interest_rate = nominal_interest_rate - inflation_expectations
    
    kappa_inputs = None
    if use_kappa:
        kappa_inputs = pd.DataFrame({
            "name": ["kappa2020Q2-Q4", "kappa2021", "kappa2022"],
            "year": [2020, 2021, 2022],
            "T.start": [np.nan, np.nan, np.nan], "T.end": [np.nan, np.nan, np.nan],
            "init": [1.0, 1.0, 1.0], "lower.bound": [1.0, 1.0, 1.0], "upper.bound": [np.inf, np.inf, np.inf],
            "theta.index": [np.nan, np.nan, np.nan], "t.stat.null": [1.0, 1.0, 1.0],
        })
        for k in range(len(kappa_inputs)):
            year = kappa_inputs.loc[k, "year"]
            covid_variance_start = (year - sample_start[0]) * 4 + (1 - sample_start[1]) + 1
            kappa_inputs.loc[k, "T.start"] = max(covid_variance_start, 0)
            covid_variance_end = (year - sample_start[0]) * 4 + (4 - sample_start[1]) + 1
            kappa_inputs.loc[k, "T.end"] = max(covid_variance_end, 0)
            if year == 2020:
                kappa_inputs.loc[k, "T.start"] += 1
    
    print("=" * 60 + "\nRunning Stage 1...\n" + "=" * 60)
    out_stage1 = rstar_stage1(log_output, inflation, relative_oil_price_inflation, relative_import_price_inflation, covid_dummy,
                               sample_end, b_y_constraint, use_kappa=use_kappa, kappa_inputs=kappa_inputs.copy() if kappa_inputs is not None else None, fix_phi=fix_phi)
    
    lambda_g = median_unbiased_estimator_stage1(out_stage1.potential_smoothed)
    print(f"  lambda_g = {lambda_g:.6f}")
    
    print("=" * 60 + "\nRunning Stage 2...\n" + "=" * 60)
    out_stage2 = rstar_stage2(log_output, inflation, relative_oil_price_inflation, relative_import_price_inflation, real_interest_rate, covid_dummy,
                               lambda_g, sample_end, a_r_constraint, b_y_constraint, use_kappa=use_kappa, kappa_inputs=kappa_inputs.copy() if kappa_inputs is not None else None, fix_phi=fix_phi)
    
    lambda_z = median_unbiased_estimator_stage2(out_stage2.y, out_stage2.x, out_stage2.kappa_vec)
    print(f"  lambda_z = {lambda_z:.6f}")
    
    print("=" * 60 + "\nRunning Stage 3...\n" + "=" * 60)
    out_stage3 = rstar_stage3(log_output, inflation, relative_oil_price_inflation, relative_import_price_inflation, real_interest_rate, covid_dummy,
                               lambda_g, lambda_z, sample_end, a_r_constraint, b_y_constraint, use_kappa=use_kappa, kappa_inputs=kappa_inputs.copy() if kappa_inputs is not None else None, fix_phi=fix_phi)
    
    phi = out_stage3.theta[12]
    print(f"  phi = {phi:.6f}")
    
    t_end = len(log_output) - 8
    dates = pd.date_range(start=f"{sample_start[0]}-{sample_start[1]*3:02d}-01", periods=t_end, freq="QS")
    
    results = pd.DataFrame({
        "Date": dates,
        "rstar_filtered": out_stage3.rstar_filtered, "g_filtered": out_stage3.trend_filtered,
        "z_filtered": out_stage3.z_filtered, "output_gap_filtered": out_stage3.output_gap_filtered,
        "rstar_smoothed": out_stage3.rstar_smoothed, "g_smoothed": out_stage3.trend_smoothed,
        "z_smoothed": out_stage3.z_smoothed, "output_gap_smoothed": out_stage3.output_gap_smoothed,
    })
    
    csv_path = data_path / "lw_port_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    validation = _compare_with_reference(excel_path, results, figures_path)
    return validation


def _compare_with_reference(excel_path: Path, results: pd.DataFrame, figures_path: Path) -> Dict[str, float]:
    """Compare with published estimates and generate plots."""
    ref = pd.read_excel(excel_path, sheet_name="data", skiprows=5)
    ref = ref.rename(columns={
        "rstar": "rstar_filtered_ref", "g": "g_filtered_ref", "z": "z_filtered_ref", "Output gap": "output_gap_filtered_ref",
        "rstar.1": "rstar_smoothed_ref", "g.1": "g_smoothed_ref", "z.1": "z_smoothed_ref", "Output gap.1": "output_gap_smoothed_ref",
    })
    ref = ref.dropna(subset=["Date"]).copy()
    ref["Date"] = pd.to_datetime(ref["Date"])
    merged = results.merge(ref, on="Date", how="inner")
    
    metrics = {}
    for col in ["rstar_filtered", "g_filtered", "z_filtered", "output_gap_filtered", "rstar_smoothed", "g_smoothed", "z_smoothed", "output_gap_smoothed"]:
        ref_col = f"{col}_ref"
        if ref_col in merged.columns:
            diff = (merged[col] - merged[ref_col]).abs()
            metrics[f"{col}_max_abs_diff"] = diff.max()
            metrics[f"{col}_rmse"] = (diff ** 2).mean() ** 0.5
    
    _generate_plots(merged, figures_path)
    print("\nValidation against published estimates:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    return metrics


def _generate_plots(merged: pd.DataFrame, figures_path: Path) -> None:
    """Generate comparison plots."""
    port_color, ref_color = "#2E86AB", "#E94F37"
    metrics_list = [
        ("rstar", "r*", "rstar_filtered", "rstar_smoothed", "rstar_filtered_ref", "rstar_smoothed_ref"),
        ("g", "g (trend growth)", "g_filtered", "g_smoothed", "g_filtered_ref", "g_smoothed_ref"),
        ("z", "z (other factors)", "z_filtered", "z_smoothed", "z_filtered_ref", "z_smoothed_ref"),
        ("output_gap", "Output Gap", "output_gap_filtered", "output_gap_smoothed", "output_gap_filtered_ref", "output_gap_smoothed_ref"),
    ]
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("LW Python Port vs Published Estimates", fontsize=14, fontweight="bold")
    for row, (name, label, filt_port, smooth_port, filt_ref, smooth_ref) in enumerate(metrics_list):
        ax = axes[row, 0]
        ax.plot(merged["Date"], merged[filt_port], color=port_color, linewidth=1.5, label="Python Port")
        ax.plot(merged["Date"], merged[filt_ref], color=ref_color, linewidth=1.5, linestyle="--", label="Published")
        ax.set_ylabel(label); ax.set_title(f"{label} - Filtered"); ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(10)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        
        ax = axes[row, 1]
        ax.plot(merged["Date"], merged[smooth_port], color=port_color, linewidth=1.5, label="Python Port")
        ax.plot(merged["Date"], merged[smooth_ref], color=ref_color, linewidth=1.5, linestyle="--", label="Published")
        ax.set_ylabel(label); ax.set_title(f"{label} - Smoothed"); ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(10)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    plt.tight_layout()
    plt.savefig(figures_path / "lw_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {figures_path / 'lw_comparison.png'}")
    
    fig2, axes2 = plt.subplots(4, 2, figsize=(14, 12))
    fig2.suptitle("Differences: Python Port - Published Estimates", fontsize=14, fontweight="bold")
    for row, (name, label, filt_port, smooth_port, filt_ref, smooth_ref) in enumerate(metrics_list):
        ax = axes2[row, 0]
        diff = merged[filt_port] - merged[filt_ref]
        ax.plot(merged["Date"], diff, color="#4A4E69", linewidth=1.5)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.fill_between(merged["Date"], diff, 0, alpha=0.3, color="#4A4E69")
        ax.set_ylabel(f"Δ {label}"); ax.set_title(f"{label} - Filtered Difference"); ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(10)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        
        ax = axes2[row, 1]
        diff = merged[smooth_port] - merged[smooth_ref]
        ax.plot(merged["Date"], diff, color="#4A4E69", linewidth=1.5)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.fill_between(merged["Date"], diff, 0, alpha=0.3, color="#4A4E69")
        ax.set_ylabel(f"Δ {label}"); ax.set_title(f"{label} - Smoothed Difference"); ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(10)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    plt.tight_layout()
    plt.savefig(figures_path / "lw_differences.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Differences plot saved to {figures_path / 'lw_differences.png'}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    default_excel = script_dir.parent / "data" / "Laubach_Williams_current_estimates.xlsx"
    
    if len(sys.argv) > 1:
        excel_path = Path(sys.argv[1])
    else:
        excel_path = default_excel
    
    if not excel_path.exists():
        print(f"Error: Data file not found at {excel_path}")
        print("Usage: python lw_estimation.py [path_to_excel]")
        sys.exit(1)
    
    output_dir = script_dir / "outputs"
    metrics = run_estimation(excel_path=excel_path, output_dir=output_dir, sample_start=(1961, 1), sample_end=(2025, 2), use_kappa=True, fix_phi=None)
    
    print("\n" + "=" * 60)
    print("Estimation complete!")
    print("=" * 60)
