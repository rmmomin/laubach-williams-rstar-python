"""
Three-stage LW estimation (2023 replication code port).

Implements:
- COVID indicator (phi parameter)
- Time-varying variance (kappa)
- HP filter initialization
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, least_squares

from .kalman import KalmanResults, kalman_filter, run_kalman
from .parameters import (
    StateSpace,
    build_stage1_system,
    build_stage2_system,
    build_stage3_system,
    calculate_initial_covariance,
    PARAM_NUM_STAGE1,
    PARAM_NUM_STAGE2,
    PARAM_NUM_STAGE3,
)
from .utils import (
    PreparedInput, 
    KappaConfig,
    build_default_kappa_inputs,
    delayed_ramp, 
    hp_filter,
    r_slice,
)


@dataclass
class Stage1Result:
    theta: np.ndarray
    loglik: float
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
    states: KalmanResults
    trend_filtered: np.ndarray
    potential_filtered: np.ndarray
    output_gap_filtered: np.ndarray
    trend_smoothed: np.ndarray
    potential_smoothed: np.ndarray
    output_gap_smoothed: np.ndarray
    lambda_z: float
    kappa_vec: np.ndarray


@dataclass
class Stage3Result:
    theta: np.ndarray
    loglik: float
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
    phi: float
    kappa_inputs: List[KappaConfig]


def _original_output_gap(log_output: np.ndarray, T: int) -> np.ndarray:
    """Compute initial output gap estimate using polynomial trend."""
    total = T + 4
    x = np.column_stack([
        np.ones(total),
        np.arange(1, total + 1, dtype=float),
        delayed_ramp(total, 56),
        delayed_ramp(total, 142),
    ])
    y = log_output[4 : T + 8]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return (y - x @ beta) * 100.0


def stage1(
    data: PreparedInput,
    sample_end: Tuple[int, int],
    b_y_constraint: Optional[float] = 0.025,
    use_kappa: bool = True,
    kappa_inputs: Optional[List[KappaConfig]] = None,
    fix_phi: Optional[float] = None,
) -> Stage1Result:
    """Run stage 1 estimation with 3 state variables."""
    T = len(data.log_output) - 8
    
    if kappa_inputs is None and use_kappa:
        kappa_inputs = build_default_kappa_inputs((data.sample_start.year, data.sample_start.quarter))
    kappa_inputs = kappa_inputs or []
    
    output_gap = _original_output_gap(data.log_output, T)
    
    # IS curve with NLS
    y_is = output_gap[4:T+4]
    y_is_l1 = output_gap[3:T+3]
    y_is_l2 = output_gap[2:T+2]
    d = data.covid_indicator[8:T+8]
    d_l1 = data.covid_indicator[7:T+7]
    d_l2 = data.covid_indicator[6:T+6]
    
    if sample_end[0] >= 2020 and fix_phi is None:
        def residual(params):
            phi, a1, a2 = params
            fitted = phi * d + a1 * (y_is_l1 - phi * d_l1) + a2 * (y_is_l2 - phi * d_l2)
            return y_is - fitted
        result = least_squares(residual, [0, 0, 0], method='lm')
        phi, a1, a2 = result.x
    else:
        X = np.column_stack([y_is_l1, y_is_l2])
        b = np.linalg.lstsq(X, y_is, rcond=None)[0]
        a1, a2 = b
        phi = fix_phi if fix_phi is not None else 0.0
    
    r_is = y_is - (phi * d + a1 * (y_is_l1 - phi * d_l1) + a2 * (y_is_l2 - phi * d_l2))
    s_is = np.sqrt(np.sum(r_is**2) / max(len(r_is) - 3, 1))
    
    # Phillips curve
    y_ph = data.inflation[8:T+8]
    og_adj = y_is_l1 - phi * d_l1
    x_ph = np.column_stack([
        data.inflation[7:T+7],
        (data.inflation[6:T+6] + data.inflation[5:T+5] + data.inflation[4:T+4]) / 3.0,
        (data.inflation[3:T+3] + data.inflation[2:T+2] + data.inflation[1:T+1] + data.inflation[0:T]) / 4.0,
        og_adj,
        data.rel_oil_inflation[7:T+7],
        data.rel_import_inflation[8:T+8],
    ])
    b_ph = np.linalg.lstsq(x_ph, y_ph, rcond=None)[0]
    r_ph = y_ph - x_ph @ b_ph
    s_ph = np.sqrt(np.sum(r_ph**2) / max(len(r_ph) - 6, 1))
    
    # Initial parameters: [a_1, a_2, b_1, b_2, b_3, b_4, b_5, g, sigma_1, sigma_2, sigma_4, phi]
    initial_theta = np.array([
        a1, a2,
        b_ph[0], b_ph[1], b_ph[3], b_ph[4], b_ph[5],
        0.85,  # g (trend growth)
        s_is, s_ph, 0.5,
        phi,
    ])
    
    n_base = len(initial_theta)
    for k, kappa_cfg in enumerate(kappa_inputs):
        kappa_cfg.theta_index = n_base + k
        initial_theta = np.append(initial_theta, kappa_cfg.init)
    
    theta_lb = np.full(len(initial_theta), -np.inf)
    theta_ub = np.full(len(initial_theta), np.inf)
    
    p = PARAM_NUM_STAGE1
    if b_y_constraint is not None:
        if initial_theta[p["b_3"]] < b_y_constraint:
            initial_theta[p["b_3"]] = b_y_constraint
        theta_lb[p["b_3"]] = b_y_constraint
    
    for k, kappa_cfg in enumerate(kappa_inputs):
        idx = n_base + k
        theta_lb[idx] = kappa_cfg.lower_bound
        theta_ub[idx] = kappa_cfg.upper_bound
    
    # Data matrices (10 columns for stage 1)
    y_data = np.column_stack([
        100.0 * data.log_output[8:T+8],
        data.inflation[8:T+8],
    ])
    
    x_data = np.column_stack([
        100.0 * data.log_output[7:T+7],
        100.0 * data.log_output[6:T+6],
        data.inflation[7:T+7],
        (data.inflation[6:T+6] + data.inflation[5:T+5] + data.inflation[4:T+4]) / 3.0,
        (data.inflation[3:T+3] + data.inflation[2:T+2] + data.inflation[1:T+1] + data.inflation[0:T]) / 4.0,
        data.rel_oil_inflation[7:T+7],
        data.rel_import_inflation[8:T+8],
        data.covid_indicator[8:T+8],
        data.covid_indicator[7:T+7],
        data.covid_indicator[6:T+6],
    ])
    
    # Initial state using HP filter (3 states)
    y_og = data.log_output[4:T+8]
    g_pot = hp_filter(y_og, lamb=36000.0)
    xi0 = np.array([
        100.0 * g_pot[3],
        100.0 * g_pot[2],
        100.0 * g_pot[1],
    ])
    P0 = np.eye(3) * 0.2
    
    def objective(theta):
        try:
            ss = build_stage1_system(theta, y_data, x_data, xi0, P0, use_kappa, kappa_inputs)
            filt = kalman_filter(
                ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons,
                y_data, x_data, ss.kappa_vec
            )
            return -filt.loglik
        except (np.linalg.LinAlgError, ValueError):
            return 1e10
    
    res = minimize(
        objective, initial_theta, method="L-BFGS-B",
        bounds=list(zip(theta_lb, theta_ub)),
        options={"maxiter": 500},
    )
    
    theta_hat = res.x
    ss = build_stage1_system(theta_hat, y_data, x_data, xi0, P0, use_kappa, kappa_inputs)
    states = run_kalman(
        ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons,
        y_data, x_data, ss.kappa_vec
    )
    
    potential_filtered = states.filtered.xi_filt[:, 0] / 100.0
    output_gap_filtered = y_data[:, 0] - potential_filtered * 100.0
    potential_smoothed = states.smoothed.xi_smooth[:, 0] / 100.0
    output_gap_smoothed = y_data[:, 0] - potential_smoothed * 100.0
    
    lambda_g = median_unbiased_lambda_g(potential_smoothed)
    
    return Stage1Result(
        theta=theta_hat,
        loglik=-res.fun,
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
    sample_end: Tuple[int, int],
    a_r_constraint: Optional[float] = -0.0025,
    b_y_constraint: Optional[float] = 0.025,
    use_kappa: bool = True,
    kappa_inputs: Optional[List[KappaConfig]] = None,
    fix_phi: Optional[float] = None,
) -> Stage2Result:
    """Run stage 2 estimation with 6 state variables."""
    T = len(data.log_output) - 8
    
    if kappa_inputs is None and use_kappa:
        kappa_inputs = build_default_kappa_inputs((data.sample_start.year, data.sample_start.quarter))
    kappa_inputs = kappa_inputs or []
    
    output_gap = _original_output_gap(data.log_output, T)
    
    # IS curve with NLS
    y_is = output_gap[4:T+4]
    y_is_l1 = output_gap[3:T+3]
    y_is_l2 = output_gap[2:T+2]
    d = data.covid_indicator[8:T+8]
    d_l1 = data.covid_indicator[7:T+7]
    d_l2 = data.covid_indicator[6:T+6]
    ir_is = (data.real_interest_rate[7:T+7] + data.real_interest_rate[6:T+6]) / 2.0
    
    if sample_end[0] >= 2020 and fix_phi is None:
        def residual(params):
            phi, a1, a2, a_r, a0 = params
            fitted = (phi * d + a1 * (y_is_l1 - phi * d_l1) + a2 * (y_is_l2 - phi * d_l2) +
                     a_r * ir_is + a0)
            return y_is - fitted
        result = least_squares(residual, [0, 0, 0, 0, 0], method='lm')
        phi, a1, a2, a_r, a0 = result.x
    else:
        X = np.column_stack([y_is_l1, y_is_l2, ir_is, np.ones(T)])
        b = np.linalg.lstsq(X, y_is, rcond=None)[0]
        a1, a2, a_r, a0 = b
        phi = fix_phi if fix_phi is not None else 0.0
    
    r_is = y_is - (phi * d + a1 * (y_is_l1 - phi * d_l1) + a2 * (y_is_l2 - phi * d_l2) + a_r * ir_is + a0)
    s_is = np.sqrt(np.sum(r_is**2) / max(len(r_is) - 5, 1))
    
    # Phillips curve
    y_ph = data.inflation[8:T+8]
    og_adj = y_is_l1 - phi * d_l1
    x_ph = np.column_stack([
        data.inflation[7:T+7],
        (data.inflation[6:T+6] + data.inflation[5:T+5] + data.inflation[4:T+4]) / 3.0,
        (data.inflation[3:T+3] + data.inflation[2:T+2] + data.inflation[1:T+1] + data.inflation[0:T]) / 4.0,
        og_adj,
        data.rel_oil_inflation[7:T+7],
        data.rel_import_inflation[8:T+8],
    ])
    b_ph = np.linalg.lstsq(x_ph, y_ph, rcond=None)[0]
    r_ph = y_ph - x_ph @ b_ph
    s_ph = np.sqrt(np.sum(r_ph**2) / max(len(r_ph) - 6, 1))
    
    # Initial parameters: [a_1, a_2, a_3, a_4, a_5, b_1, b_2, b_3, b_4, b_5, sigma_1, sigma_2, sigma_4, phi]
    initial_theta = np.array([
        a1, a2, a_r, a0,
        -a_r,  # a_5 (a_g coefficient)
        b_ph[0], b_ph[1], b_ph[3], b_ph[4], b_ph[5],
        s_is, s_ph, 0.5,
        phi,
    ])
    
    n_base = len(initial_theta)
    for k, kappa_cfg in enumerate(kappa_inputs):
        kappa_cfg.theta_index = n_base + k
        initial_theta = np.append(initial_theta, kappa_cfg.init)
    
    theta_lb = np.full(len(initial_theta), -np.inf)
    theta_ub = np.full(len(initial_theta), np.inf)
    
    p = PARAM_NUM_STAGE2
    if b_y_constraint is not None:
        if initial_theta[p["b_3"]] < b_y_constraint:
            initial_theta[p["b_3"]] = b_y_constraint
        theta_lb[p["b_3"]] = b_y_constraint
    
    if a_r_constraint is not None:
        if initial_theta[p["a_3"]] > a_r_constraint:
            initial_theta[p["a_3"]] = a_r_constraint
        theta_ub[p["a_3"]] = a_r_constraint
    
    for k, kappa_cfg in enumerate(kappa_inputs):
        idx = n_base + k
        theta_lb[idx] = kappa_cfg.lower_bound
        theta_ub[idx] = kappa_cfg.upper_bound
    
    # Data matrices (13 columns for stage 2)
    y_data = np.column_stack([
        100.0 * data.log_output[8:T+8],
        data.inflation[8:T+8],
    ])
    
    x_data = np.column_stack([
        100.0 * data.log_output[7:T+7],
        100.0 * data.log_output[6:T+6],
        data.real_interest_rate[7:T+7],
        data.real_interest_rate[6:T+6],
        data.inflation[7:T+7],
        (data.inflation[6:T+6] + data.inflation[5:T+5] + data.inflation[4:T+4]) / 3.0,
        (data.inflation[3:T+3] + data.inflation[2:T+2] + data.inflation[1:T+1] + data.inflation[0:T]) / 4.0,
        data.rel_oil_inflation[7:T+7],
        data.rel_import_inflation[8:T+8],
        np.ones(T),  # constant
        data.covid_indicator[8:T+8],
        data.covid_indicator[7:T+7],
        data.covid_indicator[6:T+6],
    ])
    
    # Initial state using HP filter (6 states)
    y_og = data.log_output[4:T+8]
    g_pot = hp_filter(y_og, lamb=36000.0)
    g_pot_diff = np.diff(g_pot)
    xi0 = np.array([
        100.0 * g_pot[3],
        100.0 * g_pot[2],
        100.0 * g_pot[1],
        100.0 * g_pot_diff[2],
        100.0 * g_pot_diff[1],
        100.0 * g_pot_diff[0],
    ])
    P0 = np.eye(6) * 0.2
    
    def objective(theta):
        try:
            ss = build_stage2_system(theta, y_data, x_data, lambda_g, xi0, P0, use_kappa, kappa_inputs)
            filt = kalman_filter(
                ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons,
                y_data, x_data, ss.kappa_vec
            )
            return -filt.loglik
        except (np.linalg.LinAlgError, ValueError):
            return 1e10
    
    res = minimize(
        objective, initial_theta, method="L-BFGS-B",
        bounds=list(zip(theta_lb, theta_ub)),
        options={"maxiter": 500},
    )
    
    theta_hat = res.x
    ss = build_stage2_system(theta_hat, y_data, x_data, lambda_g, xi0, P0, use_kappa, kappa_inputs)
    states = run_kalman(
        ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons,
        y_data, x_data, ss.kappa_vec
    )
    
    trend_filtered = states.filtered.xi_filt[:, 3] * 4.0
    potential_filtered = states.filtered.xi_filt[:, 0] / 100.0
    output_gap_filtered = y_data[:, 0] - potential_filtered * 100.0
    
    trend_smoothed = states.smoothed.xi_smooth[:, 3] * 4.0
    potential_smoothed = states.smoothed.xi_smooth[:, 0] / 100.0
    output_gap_smoothed = y_data[:, 0] - potential_smoothed * 100.0
    
    lambda_z = median_unbiased_lambda_z(output_gap_smoothed, states.smoothed.xi_smooth[:, 3], ss.kappa_vec)
    
    return Stage2Result(
        theta=theta_hat,
        loglik=-res.fun,
        states=states,
        trend_filtered=trend_filtered,
        potential_filtered=potential_filtered,
        output_gap_filtered=output_gap_filtered,
        trend_smoothed=trend_smoothed,
        potential_smoothed=potential_smoothed,
        output_gap_smoothed=output_gap_smoothed,
        lambda_z=lambda_z,
        kappa_vec=ss.kappa_vec,
    )


def stage3(
    data: PreparedInput,
    lambda_g: float,
    lambda_z: float,
    sample_end: Tuple[int, int],
    a_r_constraint: Optional[float] = -0.0025,
    b_y_constraint: Optional[float] = 0.025,
    use_kappa: bool = True,
    kappa_inputs: Optional[List[KappaConfig]] = None,
    fix_phi: Optional[float] = None,
) -> Stage3Result:
    """Run stage 3 estimation with 9 state variables."""
    T = len(data.log_output) - 8
    
    if kappa_inputs is None and use_kappa:
        kappa_inputs = build_default_kappa_inputs((data.sample_start.year, data.sample_start.quarter))
    kappa_inputs = kappa_inputs or []
    
    output_gap = _original_output_gap(data.log_output, T)
    
    # IS curve with NLS
    y_is = output_gap[4:T+4]
    y_is_l1 = output_gap[3:T+3]
    y_is_l2 = output_gap[2:T+2]
    d = data.covid_indicator[8:T+8]
    d_l1 = data.covid_indicator[7:T+7]
    d_l2 = data.covid_indicator[6:T+6]
    ir_is = (data.real_interest_rate[7:T+7] + data.real_interest_rate[6:T+6]) / 2.0
    
    if sample_end[0] >= 2020 and fix_phi is None:
        def residual(params):
            phi, a1, a2, a_r, a0 = params
            fitted = (phi * d + a1 * (y_is_l1 - phi * d_l1) + a2 * (y_is_l2 - phi * d_l2) +
                     a_r * ir_is + a0)
            return y_is - fitted
        result = least_squares(residual, [0, 0, 0, 0, 0], method='lm')
        phi, a1, a2, a_r, a0 = result.x
    else:
        X = np.column_stack([y_is_l1, y_is_l2, ir_is, np.ones(T)])
        b = np.linalg.lstsq(X, y_is, rcond=None)[0]
        a1, a2, a_r, a0 = b
        phi = fix_phi if fix_phi is not None else 0.0
    
    r_is = y_is - (phi * d + a1 * (y_is_l1 - phi * d_l1) + a2 * (y_is_l2 - phi * d_l2) + a_r * ir_is + a0)
    s_is = np.sqrt(np.sum(r_is**2) / max(len(r_is) - 5, 1))
    
    # Phillips curve
    y_ph = data.inflation[8:T+8]
    og_adj = y_is_l1 - phi * d_l1
    x_ph = np.column_stack([
        data.inflation[7:T+7],
        (data.inflation[6:T+6] + data.inflation[5:T+5] + data.inflation[4:T+4]) / 3.0,
        (data.inflation[3:T+3] + data.inflation[2:T+2] + data.inflation[1:T+1] + data.inflation[0:T]) / 4.0,
        og_adj,
        data.rel_oil_inflation[7:T+7],
        data.rel_import_inflation[8:T+8],
    ])
    b_ph = np.linalg.lstsq(x_ph, y_ph, rcond=None)[0]
    r_ph = y_ph - x_ph @ b_ph
    s_ph = np.sqrt(np.sum(r_ph**2) / max(len(r_ph) - 6, 1))
    
    # Initial parameters: [a_1, a_2, a_3, b_1, b_2, b_3, b_4, b_5, c, sigma_1, sigma_2, sigma_4, phi]
    initial_theta = np.array([
        a1, a2, a_r,
        b_ph[0], b_ph[1], b_ph[3], b_ph[4], b_ph[5],
        1.0,  # c
        s_is, s_ph, 0.7,
        phi,
    ])
    
    n_base = len(initial_theta)
    for k, kappa_cfg in enumerate(kappa_inputs):
        kappa_cfg.theta_index = n_base + k
        initial_theta = np.append(initial_theta, kappa_cfg.init)
    
    theta_lb = np.full(len(initial_theta), -np.inf)
    theta_ub = np.full(len(initial_theta), np.inf)
    
    p = PARAM_NUM_STAGE3
    if b_y_constraint is not None:
        if initial_theta[p["b_3"]] < b_y_constraint:
            initial_theta[p["b_3"]] = b_y_constraint
        theta_lb[p["b_3"]] = b_y_constraint
    
    if a_r_constraint is not None:
        if initial_theta[p["a_3"]] > a_r_constraint:
            initial_theta[p["a_3"]] = a_r_constraint
        theta_ub[p["a_3"]] = a_r_constraint
    
    if fix_phi is not None:
        theta_lb[p["phi"]] = fix_phi
        theta_ub[p["phi"]] = fix_phi
    
    for k, kappa_cfg in enumerate(kappa_inputs):
        idx = n_base + k
        theta_lb[idx] = kappa_cfg.lower_bound
        theta_ub[idx] = kappa_cfg.upper_bound
    
    # Data matrices (12 columns for stage 3)
    y_data = np.column_stack([
        100.0 * data.log_output[8:T+8],
        data.inflation[8:T+8],
    ])
    
    x_data = np.column_stack([
        100.0 * data.log_output[7:T+7],
        100.0 * data.log_output[6:T+6],
        data.real_interest_rate[7:T+7],
        data.real_interest_rate[6:T+6],
        data.inflation[7:T+7],
        (data.inflation[6:T+6] + data.inflation[5:T+5] + data.inflation[4:T+4]) / 3.0,
        (data.inflation[3:T+3] + data.inflation[2:T+2] + data.inflation[1:T+1] + data.inflation[0:T]) / 4.0,
        data.rel_oil_inflation[7:T+7],
        data.rel_import_inflation[8:T+8],
        data.covid_indicator[8:T+8],
        data.covid_indicator[7:T+7],
        data.covid_indicator[6:T+6],
    ])
    
    # Initial state using HP filter (9 states)
    y_og = data.log_output[4:T+8]
    g_pot = hp_filter(y_og, lamb=36000.0)
    g_pot_diff = np.diff(g_pot)
    xi0 = np.array([
        100.0 * g_pot[3],
        100.0 * g_pot[2],
        100.0 * g_pot[1],
        100.0 * g_pot_diff[2],
        100.0 * g_pot_diff[1],
        100.0 * g_pot_diff[0],
        0.0, 0.0, 0.0,  # z starts at 0
    ])
    
    # Calculate P0
    P0 = calculate_initial_covariance(
        initial_theta, list(zip(theta_lb, theta_ub)),
        y_data, x_data, stage=3,
        lambda_g=lambda_g, lambda_z=lambda_z, xi0=xi0,
        use_kappa=use_kappa, kappa_inputs=kappa_inputs,
    )
    
    def objective(theta):
        try:
            ss = build_stage3_system(theta, y_data, x_data, lambda_g, lambda_z, xi0, P0, use_kappa, kappa_inputs)
            filt = kalman_filter(
                ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons,
                y_data, x_data, ss.kappa_vec
            )
            return -filt.loglik
        except (np.linalg.LinAlgError, ValueError):
            return 1e10
    
    res = minimize(
        objective, initial_theta, method="L-BFGS-B",
        bounds=list(zip(theta_lb, theta_ub)),
        options={"maxiter": 5000},
    )
    
    theta_hat = res.x
    ss = build_stage3_system(theta_hat, y_data, x_data, lambda_g, lambda_z, xi0, P0, use_kappa, kappa_inputs)
    states = run_kalman(
        ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons,
        y_data, x_data, ss.kappa_vec
    )
    
    phi_final = theta_hat[p["phi"]]
    covid_adj = phi_final * data.covid_indicator[8:T+8]
    
    # Filtered estimates
    trend_filtered = states.filtered.xi_filt[:, 3] * 4.0
    z_filtered = states.filtered.xi_filt[:, 6]
    rstar_filtered = trend_filtered * theta_hat[p["c"]] + z_filtered
    potential_filtered = states.filtered.xi_filt[:, 0] / 100.0
    output_gap_filtered = y_data[:, 0] - potential_filtered * 100.0 - covid_adj
    
    # Smoothed estimates
    trend_smoothed = states.smoothed.xi_smooth[:, 3] * 4.0
    z_smoothed = states.smoothed.xi_smooth[:, 6]
    rstar_smoothed = trend_smoothed * theta_hat[p["c"]] + z_smoothed
    potential_smoothed = states.smoothed.xi_smooth[:, 0] / 100.0
    output_gap_smoothed = y_data[:, 0] - potential_smoothed * 100.0 - covid_adj
    
    return Stage3Result(
        theta=theta_hat,
        loglik=-res.fun,
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
        phi=phi_final,
        kappa_inputs=kappa_inputs,
    )


# Median unbiased estimator functions

def median_unbiased_lambda_g(series: np.ndarray) -> float:
    """Median unbiased estimator for lambda_g."""
    series = np.asarray(series, dtype=float)
    T = len(series)
    y = 400.0 * np.diff(series)
    stats = []
    for i in range(4, T - 4):
        xr = np.column_stack([
            np.ones(T - 1),
            np.concatenate([np.zeros(i), np.ones(T - i - 1)]),
        ])
        xi = np.linalg.inv(xr.T @ xr)
        b = xi @ (xr.T @ y)
        s3 = np.sum((y - xr @ b) ** 2) / (T - 3)
        stats.append(b[1] / np.sqrt(s3 * xi[1, 1]))
    
    return _lambda_from_stats(np.array(stats), T - 1)


def median_unbiased_lambda_z(output_gap: np.ndarray, g: np.ndarray, kappa_vec: np.ndarray) -> float:
    """Median unbiased estimator for lambda_z."""
    T = len(output_gap)
    y = output_gap[2:]
    x = np.column_stack([
        output_gap[1:-1],
        output_gap[:-2],
        g[2:],  # trend growth
        np.ones(T - 2),
    ])
    
    stats = []
    for i in range(4, T - 6):
        extra = np.concatenate([np.zeros(i), np.ones(T - 2 - i)])
        xr = np.column_stack([x, extra])
        xi = np.linalg.inv(xr.T @ xr)
        b = xi @ (xr.T @ y)
        s3 = np.sum((y - xr @ b) ** 2) / (T - 2 - xr.shape[1])
        stats.append(b[-1] / np.sqrt(s3 * xi[-1, -1]))
    
    return _lambda_from_stats(np.array(stats), T - 2)


VAL_EW = np.array([
    0.426, 0.476, 0.516, 0.661, 0.826, 1.111, 1.419, 1.762, 2.355, 2.91,
    3.413, 3.868, 4.925, 5.684, 6.670, 7.690, 8.477, 9.191, 10.693, 12.024,
    13.089, 14.440, 16.191, 17.332, 18.699, 20.464, 21.667, 23.851, 25.538,
    26.762, 27.874,
])

VAL_MW = np.array([
    0.689, 0.757, 0.806, 1.015, 1.234, 1.632, 2.018, 2.390, 3.081, 3.699,
    4.222, 4.776, 5.767, 6.586, 7.703, 8.683, 9.467, 10.101, 11.639, 13.039,
    13.900, 15.214, 16.806, 18.330, 19.020, 20.562, 21.837, 24.350, 26.248,
    27.089, 27.758,
])

VAL_QL = np.array([
    3.198, 3.416, 3.594, 4.106, 4.848, 5.689, 6.682, 7.626, 9.16, 10.66,
    11.841, 13.098, 15.451, 17.094, 19.423, 21.682, 23.342, 24.920, 28.174,
    30.736, 33.313, 36.109, 39.673, 41.955, 45.056, 48.647, 50.983, 55.514,
    59.278, 61.311, 64.016,
])


def _interpolate_table(stat: float, table: np.ndarray) -> Optional[float]:
    if stat <= table[0]:
        return 0.0
    for idx in range(len(table) - 1):
        if table[idx] < stat <= table[idx + 1]:
            return idx + (stat - table[idx]) / (table[idx + 1] - table[idx])
    return None


def _lambda_from_stats(stats: np.ndarray, denominator: float) -> float:
    stats = np.asarray(stats, dtype=float)
    ew = np.log(np.mean(np.exp(np.clip(stats**2 / 2.0, -500, 500))))
    mw = np.mean(stats**2)
    qlr = np.max(stats**2)
    
    lame = _interpolate_table(ew, VAL_EW)
    lamm = _interpolate_table(mw, VAL_MW)
    lamq = _interpolate_table(qlr, VAL_QL)
    
    if lame is None or lamm is None or lamq is None:
        return 0.05 / denominator
    
    return lame / denominator
