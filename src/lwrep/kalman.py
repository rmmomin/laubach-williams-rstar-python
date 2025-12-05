"""
Kalman filter and smoother for LW 2023 replication.
Supports time-varying variance via kappa vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class KalmanFilterOutput:
    xi_pred: np.ndarray
    cov_pred: np.ndarray
    xi_filt: np.ndarray
    cov_filt: np.ndarray
    loglik: float
    prediction_error: np.ndarray
    kalman_gain: np.ndarray


@dataclass
class KalmanSmootherOutput:
    xi_smooth: np.ndarray
    cov_smooth: np.ndarray


@dataclass
class KalmanResults:
    filtered: KalmanFilterOutput
    smoothed: KalmanSmootherOutput


def kalman_filter(
    xi0: np.ndarray,
    P0: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    cons: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    kappa_vec: Optional[np.ndarray] = None,
) -> KalmanFilterOutput:
    """
    Run the Kalman filter for the linear Gaussian state-space model.
    
    Supports time-varying variance via kappa_vec which scales R.

    The measurement equation follows the R convention:
        y_t = A' x_t + H' xi_t + eps_t
    where Var(eps_t) = kappa_t^2 * R
    """

    T, n_obs = y.shape
    n_state = xi0.size

    if kappa_vec is None:
        kappa_vec = np.ones(T)

    xi_pred = np.zeros((T, n_state))
    cov_pred = np.zeros((T, n_state, n_state))
    xi_filt = np.zeros_like(xi_pred)
    cov_filt = np.zeros_like(cov_pred)
    prediction_error = np.zeros((T, n_obs))
    kalman_gain = np.zeros((T, n_state, n_obs))

    xi = xi0.copy()
    P = P0.copy()
    loglik = 0.0

    for t in range(T):
        # Prediction step
        xi_m = F @ xi + cons
        P_m = F @ P @ F.T + Q

        xi_pred[t] = xi_m
        cov_pred[t] = P_m

        # Time-varying observation noise
        R_t = (kappa_vec[t] ** 2) * R

        # Innovation
        innov = y[t] - (A.T @ x[t] + H.T @ xi_m)
        prediction_error[t] = innov
        
        S = H.T @ P_m @ H + R_t
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            raise np.linalg.LinAlgError("Innovation covariance not PD")
        S_inv_innov = np.linalg.solve(S, innov)
        loglik += -0.5 * (n_obs * np.log(2.0 * np.pi) + logdet + innov.T @ S_inv_innov)

        # Update step
        K = P_m @ H @ np.linalg.inv(S)
        kalman_gain[t] = K
        xi = xi_m + K @ innov
        P = P_m - K @ H.T @ P_m

        xi_filt[t] = xi
        cov_filt[t] = P

    return KalmanFilterOutput(
        xi_pred=xi_pred,
        cov_pred=cov_pred,
        xi_filt=xi_filt,
        cov_filt=cov_filt,
        loglik=float(loglik),
        prediction_error=prediction_error,
        kalman_gain=kalman_gain,
    )


def kalman_smoother(filter_out: KalmanFilterOutput, F: np.ndarray) -> KalmanSmootherOutput:
    """Rauch-Tung-Striebel smoother."""
    xi_pred, cov_pred = filter_out.xi_pred, filter_out.cov_pred
    xi_filt, cov_filt = filter_out.xi_filt, filter_out.cov_filt

    T, n_state = xi_filt.shape
    xi_smooth = np.zeros_like(xi_filt)
    cov_smooth = np.zeros_like(cov_filt)

    xi_smooth[-1] = xi_filt[-1]
    cov_smooth[-1] = cov_filt[-1]

    for t in range(T - 2, -1, -1):
        P_f = cov_filt[t]
        P_pred_next = cov_pred[t + 1]
        J = P_f @ F.T @ np.linalg.inv(P_pred_next)
        xi_smooth[t] = xi_filt[t] + J @ (xi_smooth[t + 1] - xi_pred[t + 1])
        cov_smooth[t] = P_f + J @ (cov_smooth[t + 1] - P_pred_next) @ J.T

    return KalmanSmootherOutput(xi_smooth=xi_smooth, cov_smooth=cov_smooth)


def run_kalman(
    xi0: np.ndarray,
    P0: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    cons: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    kappa_vec: Optional[np.ndarray] = None,
) -> KalmanResults:
    filtered = kalman_filter(xi0, P0, F, Q, A, H, R, cons, y, x, kappa_vec)
    smoothed = kalman_smoother(filtered, F)
    return KalmanResults(filtered=filtered, smoothed=smoothed)
