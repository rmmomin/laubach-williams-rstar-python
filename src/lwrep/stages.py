from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .kalman import KalmanResults, kalman_filter, run_kalman
from .parameters import (
    StateSpace,
    build_stage1_system,
    build_stage2_system,
    build_stage3_system,
    calculate_covariance,
)
from .utils import PreparedInput, delayed_ramp, r_slice


@dataclass
class Stage1Result:
    theta: np.ndarray
    loglik: float
    state_space: StateSpace
    states: KalmanResults
    potential_filtered: np.ndarray
    output_gap_filtered: np.ndarray
    potential_smoothed: np.ndarray
    output_gap_smoothed: np.ndarray
    lambda_g: float


@dataclass
class Stage2Result:
    theta: np.ndarray
    loglik: float
    state_space: StateSpace
    states: KalmanResults
    trend_filtered: np.ndarray
    potential_filtered: np.ndarray
    output_gap_filtered: np.ndarray
    trend_smoothed: np.ndarray
    potential_smoothed: np.ndarray
    output_gap_smoothed: np.ndarray
    lambda_z_input_y: np.ndarray
    lambda_z_input_x: np.ndarray
    lambda_z: float


@dataclass
class Stage3Result:
    theta: np.ndarray
    loglik: float
    state_space: StateSpace
    states: KalmanResults
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
    gls: bool


def _original_output_gap(log_output: np.ndarray) -> np.ndarray:
    T = len(log_output) - 8
    total = T + 4
    x = np.column_stack(
        [
            np.ones(total),
            np.arange(1, total + 1, dtype=float),
            delayed_ramp(total, 56),
            delayed_ramp(total, 142),
        ]
    )
    y = log_output[4 : T + 8]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return (y - x @ beta) * 100.0


def _build_shared_arrays(data: PreparedInput) -> Tuple[int, np.ndarray]:
    T = len(data.log_output) - 8
    output_gap = _original_output_gap(data.log_output)
    return T, output_gap


def _kalman_loglik(
    builder,
    theta: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    **kwargs,
) -> float:
    ss = builder(theta, y_data, x_data, **kwargs)
    filtered = kalman_filter(
        ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons, y_data, x_data
    )
    return filtered.loglik


def stage1(
    data: PreparedInput,
    b_y_constraint: Optional[float] = None,
) -> Stage1Result:
    T, output_gap = _build_shared_arrays(data)

    def og_slice(start: int, end: int) -> np.ndarray:
        return r_slice(output_gap, start, end)

    def inf_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.inflation, start, end)

    def rel_oil_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.rel_oil_inflation, start, end)

    def rel_import_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.rel_import_inflation, start, end)

    def log_output_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.log_output, start, end)

    y_is = og_slice(5, T + 4)
    x_is = np.column_stack([og_slice(4, T + 3), og_slice(3, T + 2)])
    b_is = np.linalg.lstsq(x_is, y_is, rcond=None)[0]
    residual_is = y_is - x_is @ b_is
    s_is = np.sqrt(np.sum(residual_is**2) / (len(residual_is) - x_is.shape[1]))

    y_ph = inf_slice(9, T + 8)
    x_ph = np.column_stack(
        [
            inf_slice(8, T + 7),
            (inf_slice(7, T + 6) + inf_slice(6, T + 5) + inf_slice(5, T + 4)) / 3.0,
            (
                inf_slice(4, T + 3)
                + inf_slice(3, T + 2)
                + inf_slice(2, T + 1)
                + inf_slice(1, T)
            )
            / 4.0,
            og_slice(4, T + 3),
            rel_oil_slice(8, T + 7),
            rel_import_slice(9, T + 8),
        ]
    )
    b_ph = np.linalg.lstsq(x_ph, y_ph, rcond=None)[0]
    residual_ph = y_ph - x_ph @ b_ph
    s_ph = np.sqrt(np.sum(residual_ph**2) / (T - x_ph.shape[1]))

    y_data = np.column_stack(
        [
            100.0 * log_output_slice(9, T + 8),
            inf_slice(9, T + 8),
        ]
    )
    x_data = np.column_stack(
        [
            100.0 * log_output_slice(8, T + 7),
            100.0 * log_output_slice(7, T + 6),
            inf_slice(8, T + 7),
            (inf_slice(7, T + 6) + inf_slice(6, T + 5) + inf_slice(5, T + 4)) / 3.0,
            (
                inf_slice(4, T + 3)
                + inf_slice(3, T + 2)
                + inf_slice(2, T + 1)
                + inf_slice(1, T)
            )
            / 4.0,
            rel_oil_slice(8, T + 7),
            rel_import_slice(9, T + 8),
        ]
    )

    initial_theta = np.array(
        [
            b_is[0],
            b_is[1],
            b_ph[0],
            b_ph[1],
            b_ph[3],
            b_ph[4],
            b_ph[5],
            0.85,
            s_is,
            s_ph,
            0.5,
        ],
        dtype=float,
    )

    theta_lb = np.full_like(initial_theta, -np.inf)
    theta_ub = np.full_like(initial_theta, np.inf)
    if b_y_constraint is not None:
        initial_theta[4] = max(initial_theta[4], b_y_constraint)
        theta_lb[4] = b_y_constraint

    bounds = list(zip(theta_lb, theta_ub))

    def objective(theta: np.ndarray) -> float:
        return -_kalman_loglik(build_stage1_system, theta, y_data, x_data)

    res = minimize(
        objective,
        initial_theta,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"Stage 1 optimization failed: {res.message}")

    theta_hat = res.x
    ss = build_stage1_system(theta_hat, y_data, x_data)
    states = run_kalman(
        ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons, y_data, x_data
    )

    potential_filtered = states.filtered.xi_filt[:, 0] / 100.0
    output_gap_filtered = y_data[:, 0] - potential_filtered * 100.0
    potential_smoothed = states.smoothed.xi_smooth[:, 0] / 100.0
    output_gap_smoothed = y_data[:, 0] - potential_smoothed * 100.0

    lambda_g = median_unbiased_lambda_g(potential_smoothed)

    return Stage1Result(
        theta=theta_hat,
        loglik=-res.fun,
        state_space=ss,
        states=states,
        potential_filtered=potential_filtered,
        output_gap_filtered=output_gap_filtered,
        potential_smoothed=potential_smoothed,
        output_gap_smoothed=output_gap_smoothed,
        lambda_g=lambda_g,
    )


def stage2(
    data: PreparedInput,
    lambda_g: float,
    a_r_constraint: Optional[float] = None,
    b_y_constraint: Optional[float] = None,
) -> Stage2Result:
    T, output_gap = _build_shared_arrays(data)

    def og_slice(start: int, end: int) -> np.ndarray:
        return r_slice(output_gap, start, end)

    def inf_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.inflation, start, end)

    def rel_oil_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.rel_oil_inflation, start, end)

    def rel_import_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.rel_import_inflation, start, end)

    def log_output_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.log_output, start, end)

    def real_rate_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.real_interest_rate, start, end)

    y_is = og_slice(5, T + 4)
    x_is = np.column_stack(
        [
            og_slice(4, T + 3),
            og_slice(3, T + 2),
            (real_rate_slice(8, T + 7) + real_rate_slice(7, T + 6)) / 2.0,
            np.ones(T),
        ]
    )
    b_is = np.linalg.lstsq(x_is, y_is, rcond=None)[0]
    residual_is = y_is - x_is @ b_is
    s_is = np.sqrt(np.sum(residual_is**2) / (len(residual_is) - x_is.shape[1]))

    y_ph = inf_slice(9, T + 8)
    x_ph = np.column_stack(
        [
            inf_slice(8, T + 7),
            (inf_slice(7, T + 6) + inf_slice(6, T + 5) + inf_slice(5, T + 4)) / 3.0,
            (
                inf_slice(4, T + 3)
                + inf_slice(3, T + 2)
                + inf_slice(2, T + 1)
                + inf_slice(1, T)
            )
            / 4.0,
            og_slice(4, T + 3),
            rel_oil_slice(8, T + 7),
            rel_import_slice(9, T + 8),
        ]
    )
    b_ph = np.linalg.lstsq(x_ph, y_ph, rcond=None)[0]
    residual_ph = y_ph - x_ph @ b_ph
    s_ph = np.sqrt(np.sum(residual_ph**2) / (T - x_ph.shape[1]))

    y_data = np.column_stack(
        [
            100.0 * log_output_slice(9, T + 8),
            inf_slice(9, T + 8),
        ]
    )
    x_data = np.column_stack(
        [
            100.0 * log_output_slice(8, T + 7),
            100.0 * log_output_slice(7, T + 6),
            real_rate_slice(8, T + 7),
            real_rate_slice(7, T + 6),
            inf_slice(8, T + 7),
            (inf_slice(7, T + 6) + inf_slice(6, T + 5) + inf_slice(5, T + 4)) / 3.0,
            (
                inf_slice(4, T + 3)
                + inf_slice(3, T + 2)
                + inf_slice(2, T + 1)
                + inf_slice(1, T)
            )
            / 4.0,
            rel_oil_slice(8, T + 7),
            rel_import_slice(9, T + 8),
            np.ones(T),
        ]
    )

    initial_theta = np.array(
        [
            b_is[0],
            b_is[1],
            b_is[2],
            b_is[3],
            -b_is[2],
            b_ph[0],
            b_ph[1],
            b_ph[3],
            b_ph[4],
            b_ph[5],
            s_is,
            s_ph,
            0.5,
        ],
        dtype=float,
    )

    theta_lb = np.full_like(initial_theta, -np.inf)
    theta_ub = np.full_like(initial_theta, np.inf)

    if b_y_constraint is not None:
        if initial_theta[7] < b_y_constraint:
            initial_theta[7] = b_y_constraint
        theta_lb[7] = b_y_constraint

    if a_r_constraint is not None:
        if initial_theta[2] > a_r_constraint:
            initial_theta[2] = a_r_constraint
        theta_ub[2] = a_r_constraint

    bounds = list(zip(theta_lb, theta_ub))

    def objective(theta: np.ndarray) -> float:
        return -_kalman_loglik(
            build_stage2_system,
            theta,
            y_data,
            x_data,
            lambda_g=lambda_g,
        )

    res = minimize(
        objective,
        initial_theta,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"Stage 2 optimization failed: {res.message}")

    theta_hat = res.x
    ss = build_stage2_system(theta_hat, y_data, x_data, lambda_g)
    states = run_kalman(
        ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons, y_data, x_data
    )

    trend_filtered = states.filtered.xi_filt[:, 3] * 4.0
    potential_filtered = states.filtered.xi_filt[:, 0] / 100.0
    output_gap_filtered = y_data[:, 0] - potential_filtered * 100.0

    trend_smoothed = states.smoothed.xi_smooth[:, 3] * 4.0
    prefix = states.smoothed.xi_smooth[0, 2:0:-1] / 100.0
    potential_smoothed = np.concatenate(
        [prefix, states.smoothed.xi_smooth[:, 0] / 100.0]
    )
    log_output_segment = 100.0 * log_output_slice(7, T + 8)
    output_gap_smoothed = log_output_segment - potential_smoothed

    lambda_z_input_y = output_gap_smoothed[2:]
    lambda_z_input_x = np.column_stack(
        [
            output_gap_smoothed[1:-1],
            output_gap_smoothed[:-2],
            (x_data[:, 2] + x_data[:, 3]) / 2.0,
            states.smoothed.xi_smooth[:, 3],
            np.ones(T),
        ]
    )
    lambda_z = median_unbiased_lambda_z(lambda_z_input_y, lambda_z_input_x)

    return Stage2Result(
        theta=theta_hat,
        loglik=-res.fun,
        state_space=ss,
        states=states,
        trend_filtered=trend_filtered,
        potential_filtered=potential_filtered,
        output_gap_filtered=output_gap_filtered,
        trend_smoothed=trend_smoothed,
        potential_smoothed=potential_smoothed,
        output_gap_smoothed=output_gap_smoothed,
        lambda_z_input_y=lambda_z_input_y,
        lambda_z_input_x=lambda_z_input_x,
        lambda_z=lambda_z,
    )


def median_unbiased_lambda_z(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    stats = []
    for i in range(4, T - 4):
        extra = np.concatenate([np.zeros(i), np.ones(T - i)])
        xr = np.column_stack([x, extra])
        xi = np.linalg.inv(xr.T @ xr)
        b = xi @ (xr.T @ y)
        s3 = np.sum((y - xr @ b) ** 2) / (T - xr.shape[1])
        stats.append(b[-1] / np.sqrt(s3 * xi[-1, -1]))
    return _lambda_from_stats(np.array(stats), T)


def stage3(
    data: PreparedInput,
    lambda_g: float,
    lambda_z: float,
    a_r_constraint: Optional[float] = None,
    b_y_constraint: Optional[float] = None,
) -> Stage3Result:
    T, output_gap = _build_shared_arrays(data)

    def og_slice(start: int, end: int) -> np.ndarray:
        return r_slice(output_gap, start, end)

    def inf_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.inflation, start, end)

    def rel_oil_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.rel_oil_inflation, start, end)

    def rel_import_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.rel_import_inflation, start, end)

    def log_output_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.log_output, start, end)

    def real_rate_slice(start: int, end: int) -> np.ndarray:
        return r_slice(data.real_interest_rate, start, end)

    y_is = og_slice(5, T + 4)
    x_is = np.column_stack(
        [
            og_slice(4, T + 3),
            og_slice(3, T + 2),
            (real_rate_slice(8, T + 7) + real_rate_slice(7, T + 6)) / 2.0,
            np.ones(T),
        ]
    )
    b_is = np.linalg.lstsq(x_is, y_is, rcond=None)[0]
    residual_is = y_is - x_is @ b_is
    s_is = np.sqrt(np.sum(residual_is**2) / (len(residual_is) - x_is.shape[1]))

    y_ph = inf_slice(9, T + 8)
    x_ph = np.column_stack(
        [
            inf_slice(8, T + 7),
            (inf_slice(7, T + 6) + inf_slice(6, T + 5) + inf_slice(5, T + 4)) / 3.0,
            (
                inf_slice(4, T + 3)
                + inf_slice(3, T + 2)
                + inf_slice(2, T + 1)
                + inf_slice(1, T)
            )
            / 4.0,
            og_slice(4, T + 3),
            rel_oil_slice(8, T + 7),
            rel_import_slice(9, T + 8),
        ]
    )
    b_ph = np.linalg.lstsq(x_ph, y_ph, rcond=None)[0]
    residual_ph = y_ph - x_ph @ b_ph
    s_ph = np.sqrt(np.sum(residual_ph**2) / (T - x_ph.shape[1]))

    y_data = np.column_stack(
        [
            100.0 * log_output_slice(9, T + 8),
            inf_slice(9, T + 8),
        ]
    )
    x_data = np.column_stack(
        [
            100.0 * log_output_slice(8, T + 7),
            100.0 * log_output_slice(7, T + 6),
            real_rate_slice(8, T + 7),
            real_rate_slice(7, T + 6),
            inf_slice(8, T + 7),
            (inf_slice(7, T + 6) + inf_slice(6, T + 5) + inf_slice(5, T + 4)) / 3.0,
            (
                inf_slice(4, T + 3)
                + inf_slice(3, T + 2)
                + inf_slice(2, T + 1)
                + inf_slice(1, T)
            )
            / 4.0,
            rel_oil_slice(8, T + 7),
            rel_import_slice(9, T + 8),
        ]
    )

    initial_theta = np.array(
        [
            b_is[0],
            b_is[1],
            b_is[2],
            b_ph[0],
            b_ph[1],
            b_ph[3],
            b_ph[4],
            b_ph[5],
            1.0,
            s_is,
            s_ph,
            0.7,
        ],
        dtype=float,
    )

    theta_lb = np.full_like(initial_theta, -np.inf)
    theta_ub = np.full_like(initial_theta, np.inf)

    if b_y_constraint is not None:
        if initial_theta[5] < b_y_constraint:
            initial_theta[5] = b_y_constraint
        theta_lb[5] = b_y_constraint

    if a_r_constraint is not None:
        if initial_theta[2] > a_r_constraint:
            initial_theta[2] = a_r_constraint
        theta_ub[2] = a_r_constraint

    xi0_gpot = _hp_initial_state(data.log_output, T)
    P0_gpot = calculate_covariance(
        initial_theta,
        list(zip(theta_lb, theta_ub)),
        y_data,
        x_data,
        stage=3,
        lambda_g=lambda_g,
        lambda_z=lambda_z,
        xi0=xi0_gpot,
    )

    def objective(theta: np.ndarray) -> float:
        ss, _ = build_stage3_system(
            theta,
            y_data,
            x_data,
            lambda_g,
            lambda_z,
            xi0=xi0_gpot,
            P0=P0_gpot,
            xi0_gpot=xi0_gpot,
            P0_gpot=P0_gpot,
        )
        return -kalman_filter(
            ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons, y_data, x_data
        ).loglik

    res = minimize(
        objective,
        initial_theta,
        method="L-BFGS-B",
        bounds=list(zip(theta_lb, theta_ub)),
        options={"maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"Stage 3 optimization failed: {res.message}")

    theta_hat = res.x
    ss, gls = build_stage3_system(
        theta_hat,
        y_data,
        x_data,
        lambda_g,
        lambda_z,
        xi0=None,
        P0=None,
        xi0_gpot=xi0_gpot,
        P0_gpot=P0_gpot,
    )
    states = run_kalman(
        ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons, y_data, x_data
    )

    trend_filtered = states.filtered.xi_filt[:, 3] * 4.0
    z_filtered = states.filtered.xi_filt[:, 5]
    rstar_filtered = trend_filtered * theta_hat[8] + z_filtered
    potential_filtered = states.filtered.xi_filt[:, 0] / 100.0
    output_gap_filtered = y_data[:, 0] - potential_filtered * 100.0

    trend_smoothed = states.smoothed.xi_smooth[:, 3] * 4.0
    z_smoothed = states.smoothed.xi_smooth[:, 5]
    rstar_smoothed = trend_smoothed * theta_hat[8] + z_smoothed
    potential_smoothed = states.smoothed.xi_smooth[:, 0] / 100.0
    output_gap_smoothed = y_data[:, 0] - potential_smoothed * 100.0

    return Stage3Result(
        theta=theta_hat,
        loglik=-res.fun,
        state_space=ss,
        states=states,
        rstar_filtered=rstar_filtered,
        trend_filtered=trend_filtered,
        z_filtered=z_filtered,
        potential_filtered=potential_filtered,
        output_gap_filtered=output_gap_filtered,
        rstar_smoothed=rstar_smoothed,
        trend_smoothed=trend_smoothed,
        z_smoothed=z_smoothed,
        potential_smoothed=potential_smoothed,
        output_gap_smoothed=output_gap_smoothed,
        gls=gls,
    )


def _hp_initial_state(log_output: np.ndarray, T: int) -> np.ndarray:
    total = T + 4
    x = np.column_stack(
        [
            np.ones(total),
            np.arange(1, total + 1, dtype=float),
            delayed_ramp(total, 56),
            delayed_ramp(total, 142),
        ]
    )
    y = log_output[4 : T + 8]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    g_pot = x @ beta
    return np.array(
        [
            100.0 * g_pot[4],
            100.0 * g_pot[3],
            100.0 * g_pot[2],
            100.0 * beta[1],
            100.0 * beta[1],
            0.0,
            0.0,
        ],
        dtype=float,
    )


def median_unbiased_lambda_g(series: np.ndarray) -> float:
    series = np.asarray(series, dtype=float)
    T = len(series)
    y = 400.0 * np.diff(series)
    stats = []
    for i in range(4, T - 4):
        xr = np.column_stack(
            [
                np.ones(T - 1),
                np.concatenate([np.zeros(i), np.ones(T - i - 1)]),
            ]
        )
        xi = np.linalg.inv(xr.T @ xr)
        b = xi @ (xr.T @ y)
        s3 = np.sum((y - xr @ b) ** 2) / (T - 3)
        stats.append(b[1] / np.sqrt(s3 * xi[1, 1]))

    return _lambda_from_stats(np.array(stats), T - 1)


def _interpolate_table(stat: float, table: np.ndarray) -> Optional[float]:
    if stat <= table[0]:
        return 0.0
    for idx in range(len(table) - 1):
        if table[idx] < stat <= table[idx + 1]:
            return idx + (stat - table[idx]) / (table[idx + 1] - table[idx])
    return None


VAL_EW = np.array(
    [
        0.426,
        0.476,
        0.516,
        0.661,
        0.826,
        1.111,
        1.419,
        1.762,
        2.355,
        2.91,
        3.413,
        3.868,
        4.925,
        5.684,
        6.670,
        7.690,
        8.477,
        9.191,
        10.693,
        12.024,
        13.089,
        14.440,
        16.191,
        17.332,
        18.699,
        20.464,
        21.667,
        23.851,
        25.538,
        26.762,
        27.874,
    ]
)
VAL_MW = np.array(
    [
        0.689,
        0.757,
        0.806,
        1.015,
        1.234,
        1.632,
        2.018,
        2.390,
        3.081,
        3.699,
        4.222,
        4.776,
        5.767,
        6.586,
        7.703,
        8.683,
        9.467,
        10.101,
        11.639,
        13.039,
        13.900,
        15.214,
        16.806,
        18.330,
        19.020,
        20.562,
        21.837,
        24.350,
        26.248,
        27.089,
        27.758,
    ]
)
VAL_QL = np.array(
    [
        3.198,
        3.416,
        3.594,
        4.106,
        4.848,
        5.689,
        6.682,
        7.626,
        9.16,
        10.66,
        11.841,
        13.098,
        15.451,
        17.094,
        19.423,
        21.682,
        23.342,
        24.920,
        28.174,
        30.736,
        33.313,
        36.109,
        39.673,
        41.955,
        45.056,
        48.647,
        50.983,
        55.514,
        59.278,
        61.311,
        64.016,
    ]
)


def _lambda_from_stats(stats: np.ndarray, denominator: float) -> float:
    stats = np.asarray(stats, dtype=float)
    ew = np.log(np.mean(np.exp(stats**2 / 2.0)))
    mw = np.mean(stats**2)
    qlr = np.max(stats**2)

    lame = _interpolate_table(ew, VAL_EW)
    lamm = _interpolate_table(mw, VAL_MW)
    lamq = _interpolate_table(qlr, VAL_QL)
    if lame is None or lamm is None or lamq is None:
        raise RuntimeError("Median-unbiased lookup failed.")
    return lame / denominator

