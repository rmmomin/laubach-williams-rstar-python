"""
Kalman filter and smoother - direct port of kalman.log.likelihood.R and kalman.states.R
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class FilteredStates:
    xi_ttm1: np.ndarray  # Predicted state (T x n_state)
    P_ttm1: np.ndarray   # Predicted covariance (T*n_state x n_state)
    xi_tt: np.ndarray    # Filtered state (T x n_state)
    P_tt: np.ndarray     # Filtered covariance (T*n_state x n_state)
    prediction_error: np.ndarray
    kalman_gain: np.ndarray


@dataclass
class SmoothedStates:
    xi_tT: np.ndarray  # Smoothed state (T x n_state)
    P_tT: np.ndarray   # Smoothed covariance (T*n_state x n_state)


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
    """
    Direct port of kalman.log.likelihood.R
    
    Returns:
        ll_vec: Log-likelihood for each period
        ll_cum: Cumulative log-likelihood  
        prediction_error: Prediction errors
    """
    t_end = y.shape[0]
    n = y.shape[1]
    
    ll_vec = np.zeros(t_end)
    ll_cum = 0.0
    prediction_error_vec = np.zeros((t_end, n))
    
    xi_tt = xi_tm1tm1.copy()
    P_tt = P_tm1tm1.copy()
    
    for t in range(t_end):
        # Predict
        xi_ttm1 = F @ xi_tt + cons.flatten()
        P_ttm1 = F @ P_tt @ F.T + Q
        
        # Prediction error
        prediction_error = y[t] - A.T @ x[t] - H.T @ xi_ttm1
        
        # Innovation covariance with kappa scaling
        HPHR = H.T @ P_ttm1 @ H + (kappa[t] ** 2) * R
        
        # Log-likelihood (using 2*pi = 4*atan(1))
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
        
        # Update
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
    """Direct port of kalman.states.filtered() - iterative version"""
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
        # Predict
        xi_ttm1 = F @ xi_tt + cons.flatten()
        P_ttm1 = F @ P_tt @ F.T + Q
        
        # Prediction error
        prediction_error = y[t] - A.T @ x[t] - H.T @ xi_ttm1
        
        # Innovation covariance
        HPHR = H.T @ P_ttm1 @ H + (kappa[t] ** 2) * R
        
        # Update
        xi_tt = xi_ttm1 + P_ttm1 @ H @ np.linalg.solve(HPHR, prediction_error)
        P_tt = P_ttm1 - P_ttm1 @ H @ np.linalg.solve(HPHR, H.T @ P_ttm1)
        kalman_gain = P_ttm1 @ H @ np.linalg.inv(HPHR)
        
        # Store
        xi_ttm1_arr[t] = xi_ttm1
        P_ttm1_arr[t * n_state:(t + 1) * n_state, :] = P_ttm1
        xi_tt_arr[t] = xi_tt
        P_tt_arr[t * n_state:(t + 1) * n_state, :] = P_tt
        prediction_error_arr[t] = prediction_error
        kalman_gain_arr[t * n_state:(t + 1) * n_state, :] = kalman_gain
    
    return FilteredStates(
        xi_ttm1=xi_ttm1_arr,
        P_ttm1=P_ttm1_arr,
        xi_tt=xi_tt_arr,
        P_tt=P_tt_arr,
        prediction_error=prediction_error_arr,
        kalman_gain=kalman_gain_arr,
    )


def kalman_states_smoothed(filtered: FilteredStates, F: np.ndarray) -> SmoothedStates:
    """Direct port of kalman.states.smoothed() - iterative version"""
    t_end = filtered.xi_tt.shape[0]
    n_state = filtered.xi_tt.shape[1]
    
    xi_tT_arr = np.zeros((t_end, n_state))
    P_tT_arr = np.zeros((t_end * n_state, n_state))
    
    # Initialize at T
    xi_tT_arr[t_end - 1] = filtered.xi_tt[t_end - 1]
    P_tT_arr[(t_end - 1) * n_state:t_end * n_state, :] = filtered.P_tt[(t_end - 1) * n_state:t_end * n_state, :]
    
    # Backward recursion
    for t in range(t_end - 2, -1, -1):
        P_tt = filtered.P_tt[t * n_state:(t + 1) * n_state, :]
        P_tp1t = filtered.P_ttm1[(t + 1) * n_state:(t + 2) * n_state, :]
        
        # Use solve instead of inv for numerical stability
        try:
            J_t = P_tt @ F.T @ np.linalg.solve(P_tp1t, np.eye(n_state))
        except np.linalg.LinAlgError:
            # Add small regularization if singular
            J_t = P_tt @ F.T @ np.linalg.solve(P_tp1t + 1e-10 * np.eye(n_state), np.eye(n_state))
        
        xi_tt = filtered.xi_tt[t]
        xi_tp1t = filtered.xi_ttm1[t + 1]
        xi_tp1T = xi_tT_arr[t + 1]
        P_tp1T = P_tT_arr[(t + 1) * n_state:(t + 2) * n_state, :]
        
        xi_tT_arr[t] = xi_tt + J_t @ (xi_tp1T - xi_tp1t)
        P_tT_arr[t * n_state:(t + 1) * n_state, :] = P_tt + J_t @ (P_tp1T - P_tp1t) @ J_t.T
    
    return SmoothedStates(xi_tT=xi_tT_arr, P_tT=P_tT_arr)


def kalman_states(
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
) -> KalmanStates:
    """Direct port of kalman.states()"""
    filtered = kalman_states_filtered(
        xi_tm1tm1, P_tm1tm1, F, Q, A, H, R, kappa, cons, y, x
    )
    smoothed = kalman_states_smoothed(filtered, F)
    return KalmanStates(filtered=filtered, smoothed=smoothed)
