"""
Stage 1, 2, 3 estimation - direct port of rstar.stage*.R
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from statsmodels.tsa.filters.hp_filter import hpfilter

from .kalman import kalman_log_likelihood, kalman_states, KalmanStates
from .parameters import (
    StateSpaceMatrices,
    unpack_parameters_stage1,
    unpack_parameters_stage2,
    unpack_parameters_stage3,
)


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
    y: np.ndarray  # For lambda_z estimation
    x: np.ndarray  # For lambda_z estimation
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


def _log_likelihood_wrapper(
    parameters: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    stage: int,
    lambda_g: Optional[float],
    lambda_z: Optional[float],
    xi_00: np.ndarray,
    P_00: np.ndarray,
    use_kappa: bool,
    kappa_inputs: Optional[pd.DataFrame],
    param_num: Dict[str, int],
) -> float:
    """Wrapper for log likelihood calculation."""
    if stage == 1:
        matrices = unpack_parameters_stage1(
            parameters, y_data, x_data, xi_00, P_00, use_kappa, kappa_inputs, param_num
        )
    elif stage == 2:
        matrices = unpack_parameters_stage2(
            parameters, y_data, x_data, lambda_g, xi_00, P_00, use_kappa, kappa_inputs, param_num
        )
    elif stage == 3:
        matrices = unpack_parameters_stage3(
            parameters, y_data, x_data, lambda_g, lambda_z, xi_00, P_00, use_kappa, kappa_inputs, param_num
        )
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    _, ll_cum, _ = kalman_log_likelihood(
        matrices.xi_00, matrices.P_00, matrices.F, matrices.Q,
        matrices.A, matrices.H, matrices.R, matrices.kappa_vec,
        matrices.cons, y_data, x_data
    )
    return ll_cum


def _kalman_states_wrapper(
    parameters: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    stage: int,
    lambda_g: Optional[float],
    lambda_z: Optional[float],
    xi_00: np.ndarray,
    P_00: np.ndarray,
    use_kappa: bool,
    kappa_inputs: Optional[pd.DataFrame],
    param_num: Dict[str, int],
) -> Tuple[KalmanStates, StateSpaceMatrices]:
    """Wrapper for Kalman states calculation."""
    if stage == 1:
        matrices = unpack_parameters_stage1(
            parameters, y_data, x_data, xi_00, P_00, use_kappa, kappa_inputs, param_num
        )
    elif stage == 2:
        matrices = unpack_parameters_stage2(
            parameters, y_data, x_data, lambda_g, xi_00, P_00, use_kappa, kappa_inputs, param_num
        )
    elif stage == 3:
        matrices = unpack_parameters_stage3(
            parameters, y_data, x_data, lambda_g, lambda_z, xi_00, P_00, use_kappa, kappa_inputs, param_num
        )
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    states = kalman_states(
        matrices.xi_00, matrices.P_00, matrices.F, matrices.Q,
        matrices.A, matrices.H, matrices.R, matrices.kappa_vec,
        matrices.cons, y_data, x_data
    )
    return states, matrices


def calculate_covariance(
    initial_parameters: np.ndarray,
    theta_lb: np.ndarray,
    theta_ub: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    stage: int,
    lambda_g: Optional[float],
    lambda_z: Optional[float],
    xi_00: np.ndarray,
    use_kappa: bool,
    kappa_inputs: Optional[pd.DataFrame],
    param_num: Dict[str, int],
) -> np.ndarray:
    """Direct port of calculate.covariance.R"""
    n_state_vars = len(xi_00)
    P_00 = np.eye(n_state_vars) * 0.2
    
    def neg_ll(theta):
        return -_log_likelihood_wrapper(
            theta, y_data, x_data, stage, lambda_g, lambda_z,
            xi_00, P_00, use_kappa, kappa_inputs, param_num
        )
    
    bounds = list(zip(theta_lb, theta_ub))
    result = minimize(
        neg_ll, initial_parameters, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 5000, "ftol": 1e-8}
    )
    
    if not result.success:
        print(f"Warning: calculate_covariance optimization did not converge: {result.message}")
    
    theta = result.x
    
    # Run Kalman filter to get initial covariance
    states, _ = _kalman_states_wrapper(
        theta, y_data, x_data, stage, lambda_g, lambda_z,
        xi_00, P_00, use_kappa, kappa_inputs, param_num
    )
    
    # Return P.ttm1 from first period
    return states.filtered.P_ttm1[:n_state_vars, :]


def rstar_stage1(
    log_output: np.ndarray,
    inflation: np.ndarray,
    relative_oil_price_inflation: np.ndarray,
    relative_import_price_inflation: np.ndarray,
    covid_dummy: np.ndarray,
    sample_end: Tuple[int, int],
    b_y_constraint: Optional[float] = None,
    xi_00: Optional[np.ndarray] = None,
    P_00: Optional[np.ndarray] = None,
    use_kappa: bool = False,
    kappa_inputs: Optional[pd.DataFrame] = None,
    fix_phi: Optional[float] = None,
) -> Stage1Result:
    """Direct port of rstar.stage1.R"""
    stage = 1
    t_end = len(log_output) - 8
    
    # Original output gap estimate
    x_og = np.column_stack([
        np.ones(t_end + 4),
        np.arange(1, t_end + 5),
        np.concatenate([np.zeros(56), np.arange(1, t_end + 4 - 56 + 1)]) if t_end + 4 > 56 else np.zeros(t_end + 4),
        np.concatenate([np.zeros(142), np.arange(1, t_end + 4 - 142 + 1)]) if t_end + 4 > 142 else np.zeros(t_end + 4),
    ])
    y_og = log_output[4:t_end + 8]
    output_gap = (y_og - x_og @ np.linalg.solve(x_og.T @ x_og, x_og.T @ y_og)) * 100
    
    # Initialize xi_00 from HP filter if not provided
    if xi_00 is None:
        print("Stage 1: xi.00 from HP trend in log output")
        g_pot, _ = hpfilter(y_og, lamb=36000)
        xi_00 = np.array([100 * g_pot[3], 100 * g_pot[2], 100 * g_pot[1]])
    else:
        print("Stage 1: Using xi.00 input")
    
    # IS curve estimation
    y_is = output_gap[4:t_end + 4]
    y_is_l1 = output_gap[3:t_end + 3]
    y_is_l2 = output_gap[2:t_end + 2]
    d = covid_dummy[8:t_end + 8]
    d_l1 = covid_dummy[7:t_end + 7]
    d_l2 = covid_dummy[6:t_end + 6]
    
    # NLS for IS curve
    if sample_end[0] >= 2020 and fix_phi is None:
        print("Stage 1 initial IS: NLS with phi")
        def is_residuals(params):
            phi, a1, a2 = params
            pred = phi * d + a1 * (y_is_l1 - phi * d_l1) + a2 * (y_is_l2 - phi * d_l2)
            return y_is - pred
        
        result = least_squares(is_residuals, [0, 0, 0])
        b_is = {"phi": result.x[0], "a_1": result.x[1], "a_2": result.x[2]}
    else:
        print("Stage 1 initial IS: OLS without phi")
        def is_residuals(params):
            a1, a2 = params
            pred = a1 * y_is_l1 + a2 * y_is_l2
            return y_is - pred
        
        result = least_squares(is_residuals, [0, 0])
        b_is = {"a_1": result.x[0], "a_2": result.x[1], "phi": 0.0}
    
    if fix_phi is not None:
        b_is["phi"] = fix_phi
    
    r_is = is_residuals([b_is.get("phi", 0), b_is["a_1"], b_is["a_2"]] if "phi" in b_is and b_is["phi"] != 0 else [b_is["a_1"], b_is["a_2"]])
    s_is = np.sqrt(np.sum(r_is ** 2) / (len(r_is) - len(b_is)))
    
    # Phillips curve
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
    
    # Initial parameters
    initial_parameters = np.array([
        b_is["a_1"], b_is["a_2"],
        b_ph[0], b_ph[1], b_ph[3], b_ph[4], b_ph[5],
        0.85,  # g
        s_is, s_ph, 0.5,  # sigmas
        b_is["phi"]
    ])
    
    param_num = {
        "a_1": 1, "a_2": 2, "b_1": 3, "b_2": 4, "b_3": 5, "b_4": 6, "b_5": 7,
        "g": 8, "sigma_1": 9, "sigma_2": 10, "sigma_4": 11, "phi": 12
    }
    
    n_params = len(initial_parameters)
    
    # Build data matrices
    y_data = np.column_stack([
        100 * log_output[8:t_end + 8],
        inflation[8:t_end + 8]
    ])
    x_data = np.column_stack([
        100 * log_output[7:t_end + 7],
        100 * log_output[6:t_end + 6],
        inflation[7:t_end + 7],
        (inflation[6:t_end + 6] + inflation[5:t_end + 5] + inflation[4:t_end + 4]) / 3,
        (inflation[3:t_end + 3] + inflation[2:t_end + 2] + inflation[1:t_end + 1] + inflation[0:t_end]) / 4,
        relative_oil_price_inflation[7:t_end + 7],
        relative_import_price_inflation[8:t_end + 8],
        covid_dummy[8:t_end + 8],
        covid_dummy[7:t_end + 7],
        covid_dummy[6:t_end + 6],
    ])
    
    # Set bounds
    theta_lb = np.full(n_params, -np.inf)
    theta_ub = np.full(n_params, np.inf)
    
    if b_y_constraint is not None:
        print(f"Setting a lower bound of b_y > {b_y_constraint} in Stage 1")
        if initial_parameters[param_num["b_3"] - 1] < b_y_constraint:
            initial_parameters[param_num["b_3"] - 1] = b_y_constraint
        theta_lb[param_num["b_3"] - 1] = b_y_constraint
    
    if fix_phi is not None:
        print(f"Fixing phi at {fix_phi}")
        theta_lb[param_num["phi"] - 1] = fix_phi
        theta_ub[param_num["phi"] - 1] = fix_phi
    
    # Add kappa parameters
    if use_kappa and kappa_inputs is not None:
        for k, row in kappa_inputs.iterrows():
            theta_ind = n_params + k
            kappa_inputs.loc[k, "theta.index"] = theta_ind + 1  # 1-indexed for R compatibility
            param_num[row["name"]] = theta_ind + 1
            initial_parameters = np.append(initial_parameters, row["init"])
            theta_lb = np.append(theta_lb, row["lower.bound"])
            theta_ub = np.append(theta_ub, row["upper.bound"])
            print(f"Initializing {row['name']} at {row['init']}")
    
    # Initialize P_00
    if P_00 is None:
        print("Stage 1: Initializing covariance matrix")
        P_00 = calculate_covariance(
            initial_parameters, theta_lb, theta_ub,
            y_data, x_data, stage, None, None, xi_00,
            use_kappa, kappa_inputs, param_num
        )
    else:
        print("Stage 1: Using P.00 input")
    
    # Optimize
    def neg_ll(theta):
        return -_log_likelihood_wrapper(
            theta, y_data, x_data, stage, None, None,
            xi_00, P_00, use_kappa, kappa_inputs, param_num
        )
    
    bounds = list(zip(theta_lb, theta_ub))
    result = minimize(
        neg_ll, initial_parameters, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 5000, "ftol": 1e-8}
    )
    
    if not result.success:
        print(f"Warning: Stage 1 optimization: {result.message}")
    else:
        print(f"Stage 1: The terminal conditions in nloptr are {result.message}")
    
    theta = result.x
    log_likelihood = -result.fun
    
    # Get states
    states, matrices = _kalman_states_wrapper(
        theta, y_data, x_data, stage, None, None,
        xi_00, P_00, use_kappa, kappa_inputs, param_num
    )
    
    # Extract estimates
    potential_filtered = states.filtered.xi_tt[:, 0] / 100
    output_gap_filtered = y_data[:, 0] - potential_filtered * 100 - theta[param_num["phi"] - 1] * covid_dummy[8:t_end + 8]
    
    potential_smoothed = states.smoothed.xi_tT[:, 0] / 100
    output_gap_smoothed = y_data[:, 0] - potential_smoothed * 100 - theta[param_num["phi"] - 1] * covid_dummy[8:t_end + 8]
    
    return Stage1Result(
        theta=theta,
        log_likelihood=log_likelihood,
        states=states,
        matrices=matrices,
        xi_00=xi_00,
        P_00=P_00,
        potential_filtered=potential_filtered,
        output_gap_filtered=output_gap_filtered,
        potential_smoothed=potential_smoothed,
        output_gap_smoothed=output_gap_smoothed,
    )


def rstar_stage2(
    log_output: np.ndarray,
    inflation: np.ndarray,
    relative_oil_price_inflation: np.ndarray,
    relative_import_price_inflation: np.ndarray,
    real_interest_rate: np.ndarray,
    covid_dummy: np.ndarray,
    lambda_g: float,
    sample_end: Tuple[int, int],
    a_r_constraint: Optional[float] = None,
    b_y_constraint: Optional[float] = None,
    xi_00: Optional[np.ndarray] = None,
    P_00: Optional[np.ndarray] = None,
    use_kappa: bool = False,
    kappa_inputs: Optional[pd.DataFrame] = None,
    fix_phi: Optional[float] = None,
) -> Stage2Result:
    """Direct port of rstar.stage2.R"""
    stage = 2
    t_end = len(log_output) - 8
    
    # Original output gap
    x_og = np.column_stack([
        np.ones(t_end + 4),
        np.arange(1, t_end + 5),
        np.concatenate([np.zeros(56), np.arange(1, t_end + 4 - 56 + 1)]) if t_end + 4 > 56 else np.zeros(t_end + 4),
        np.concatenate([np.zeros(142), np.arange(1, t_end + 4 - 142 + 1)]) if t_end + 4 > 142 else np.zeros(t_end + 4),
    ])
    y_og = log_output[4:t_end + 8]
    output_gap = (y_og - x_og @ np.linalg.solve(x_og.T @ x_og, x_og.T @ y_og)) * 100
    
    # Initialize xi_00 from HP filter
    if xi_00 is None:
        print("Stage 2: xi.00 from HP trend in log output")
        g_pot, _ = hpfilter(y_og, lamb=36000)
        g_pot_diff = np.diff(g_pot)
        xi_00 = np.array([
            100 * g_pot[3], 100 * g_pot[2], 100 * g_pot[1],
            100 * g_pot_diff[2], 100 * g_pot_diff[1], 100 * g_pot_diff[0]
        ])
    else:
        print("Stage 2: Using xi.00 input")
    
    # IS curve
    y_is = output_gap[4:t_end + 4]
    y_is_l1 = output_gap[3:t_end + 3]
    y_is_l2 = output_gap[2:t_end + 2]
    d = covid_dummy[8:t_end + 8]
    d_l1 = covid_dummy[7:t_end + 7]
    d_l2 = covid_dummy[6:t_end + 6]
    ir_is = (real_interest_rate[7:t_end + 7] + real_interest_rate[6:t_end + 6]) / 2
    
    if sample_end[0] >= 2020 and fix_phi is None:
        print("Stage 2 initial IS: NLS with phi")
        def is_residuals(params):
            phi, a1, a2, ar, a0 = params
            pred = phi * d + a1 * (y_is_l1 - phi * d_l1) + a2 * (y_is_l2 - phi * d_l2) + ar * ir_is + a0
            return y_is - pred
        
        result = least_squares(is_residuals, [0, 0, 0, 0, 0])
        b_is = {"phi": result.x[0], "a_1": result.x[1], "a_2": result.x[2], "a_r": result.x[3], "a_0": result.x[4]}
    else:
        print("Stage 2 initial IS: NLS without phi")
        def is_residuals(params):
            a1, a2, ar, a0 = params
            pred = a1 * y_is_l1 + a2 * y_is_l2 + ar * ir_is + a0
            return y_is - pred
        
        result = least_squares(is_residuals, [0, 0, 0, 0])
        b_is = {"a_1": result.x[0], "a_2": result.x[1], "a_r": result.x[2], "a_0": result.x[3], "phi": 0.0}
    
    if fix_phi is not None:
        b_is["phi"] = fix_phi
    
    r_is = y_is - (b_is["phi"] * d + b_is["a_1"] * (y_is_l1 - b_is["phi"] * d_l1) + 
                   b_is["a_2"] * (y_is_l2 - b_is["phi"] * d_l2) + b_is["a_r"] * ir_is + b_is["a_0"])
    s_is = np.sqrt(np.sum(r_is ** 2) / (len(r_is) - len(b_is)))
    
    # Phillips curve
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
    
    # Initial parameters
    initial_parameters = np.array([
        b_is["a_1"], b_is["a_2"], b_is["a_r"], b_is["a_0"], -b_is["a_r"],
        b_ph[0], b_ph[1], b_ph[3], b_ph[4], b_ph[5],
        s_is, s_ph, 0.5, b_is["phi"]
    ])
    
    param_num = {
        "a_1": 1, "a_2": 2, "a_3": 3, "a_4": 4, "a_5": 5,
        "b_1": 6, "b_2": 7, "b_3": 8, "b_4": 9, "b_5": 10,
        "sigma_1": 11, "sigma_2": 12, "sigma_4": 13, "phi": 14
    }
    
    n_params = len(initial_parameters)
    
    # Build data matrices
    y_data = np.column_stack([
        100 * log_output[8:t_end + 8],
        inflation[8:t_end + 8]
    ])
    x_data = np.column_stack([
        100 * log_output[7:t_end + 7],
        100 * log_output[6:t_end + 6],
        real_interest_rate[7:t_end + 7],
        real_interest_rate[6:t_end + 6],
        inflation[7:t_end + 7],
        (inflation[6:t_end + 6] + inflation[5:t_end + 5] + inflation[4:t_end + 4]) / 3,
        (inflation[3:t_end + 3] + inflation[2:t_end + 2] + inflation[1:t_end + 1] + inflation[0:t_end]) / 4,
        relative_oil_price_inflation[7:t_end + 7],
        relative_import_price_inflation[8:t_end + 8],
        np.ones(t_end),
        covid_dummy[8:t_end + 8],
        covid_dummy[7:t_end + 7],
        covid_dummy[6:t_end + 6],
    ])
    
    # Set bounds
    theta_lb = np.full(n_params, -np.inf)
    theta_ub = np.full(n_params, np.inf)
    
    if b_y_constraint is not None:
        print(f"Setting a lower bound of b_y > {b_y_constraint} in Stage 2")
        if initial_parameters[param_num["b_3"] - 1] < b_y_constraint:
            initial_parameters[param_num["b_3"] - 1] = b_y_constraint
        theta_lb[param_num["b_3"] - 1] = b_y_constraint
    
    if a_r_constraint is not None:
        print(f"Setting an upper bound of a_r < {a_r_constraint} in Stage 2")
        if initial_parameters[param_num["a_3"] - 1] > a_r_constraint:
            initial_parameters[param_num["a_3"] - 1] = a_r_constraint
        theta_ub[param_num["a_3"] - 1] = a_r_constraint
    
    if fix_phi is not None:
        print(f"Fixing phi at {fix_phi}")
        theta_lb[param_num["phi"] - 1] = fix_phi
        theta_ub[param_num["phi"] - 1] = fix_phi
    
    # Add kappa parameters
    if use_kappa and kappa_inputs is not None:
        for k, row in kappa_inputs.iterrows():
            theta_ind = n_params + k
            kappa_inputs.loc[k, "theta.index"] = theta_ind + 1
            param_num[row["name"]] = theta_ind + 1
            initial_parameters = np.append(initial_parameters, row["init"])
            theta_lb = np.append(theta_lb, row["lower.bound"])
            theta_ub = np.append(theta_ub, row["upper.bound"])
            print(f"Initializing {row['name']} at {row['init']}")
    
    # Initialize P_00
    if P_00 is None:
        print("Stage 2: Initializing covariance matrix")
        P_00 = calculate_covariance(
            initial_parameters, theta_lb, theta_ub,
            y_data, x_data, stage, lambda_g, None, xi_00,
            use_kappa, kappa_inputs, param_num
        )
    else:
        print("Stage 2: Using P.00 input")
    
    # Optimize
    def neg_ll(theta):
        return -_log_likelihood_wrapper(
            theta, y_data, x_data, stage, lambda_g, None,
            xi_00, P_00, use_kappa, kappa_inputs, param_num
        )
    
    bounds = list(zip(theta_lb, theta_ub))
    result = minimize(
        neg_ll, initial_parameters, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 5000, "ftol": 1e-8}
    )
    
    if not result.success:
        print(f"Warning: Stage 2 optimization: {result.message}")
    else:
        print(f"Stage 2: The terminal conditions in nloptr are {result.message}")
    
    theta = result.x
    log_likelihood = -result.fun
    
    # Get states
    states, matrices = _kalman_states_wrapper(
        theta, y_data, x_data, stage, lambda_g, None,
        xi_00, P_00, use_kappa, kappa_inputs, param_num
    )
    
    # Extract estimates
    trend_smoothed = states.smoothed.xi_tT[:, 3] * 4
    potential_smoothed_full = np.concatenate([
        states.smoothed.xi_tT[0, 2:0:-1],  # [xi[0,2], xi[0,1]]
        states.smoothed.xi_tT[:, 0]
    ])
    output_gap_smoothed = 100 * log_output[6:t_end + 8] - potential_smoothed_full - theta[param_num["phi"] - 1] * covid_dummy[6:t_end + 8]
    
    # Inputs for lambda_z estimation
    y_lambda = output_gap_smoothed[2:]
    x_lambda = np.column_stack([
        output_gap_smoothed[1:-1],
        output_gap_smoothed[:-2],
        (x_data[:, 2] + x_data[:, 3]) / 2,
        trend_smoothed,
        np.ones(t_end)
    ])
    
    trend_filtered = states.filtered.xi_tt[:, 3] * 4
    potential_filtered = states.filtered.xi_tt[:, 0] / 100
    output_gap_filtered = y_data[:, 0] - potential_filtered * 100 - theta[param_num["phi"] - 1] * covid_dummy[8:t_end + 8]
    
    return Stage2Result(
        theta=theta,
        log_likelihood=log_likelihood,
        states=states,
        matrices=matrices,
        xi_00=xi_00,
        P_00=P_00,
        y=y_lambda,
        x=x_lambda,
        kappa_vec=matrices.kappa_vec,
        trend_filtered=trend_filtered,
        potential_filtered=potential_filtered,
        output_gap_filtered=output_gap_filtered,
        trend_smoothed=trend_smoothed,
        potential_smoothed=potential_smoothed_full[2:],
        output_gap_smoothed=output_gap_smoothed[2:],
    )


def rstar_stage3(
    log_output: np.ndarray,
    inflation: np.ndarray,
    relative_oil_price_inflation: np.ndarray,
    relative_import_price_inflation: np.ndarray,
    real_interest_rate: np.ndarray,
    covid_dummy: np.ndarray,
    lambda_g: float,
    lambda_z: float,
    sample_end: Tuple[int, int],
    a_r_constraint: Optional[float] = None,
    b_y_constraint: Optional[float] = None,
    xi_00: Optional[np.ndarray] = None,
    P_00: Optional[np.ndarray] = None,
    use_kappa: bool = False,
    kappa_inputs: Optional[pd.DataFrame] = None,
    fix_phi: Optional[float] = None,
) -> Stage3Result:
    """Direct port of rstar.stage3.R"""
    stage = 3
    t_end = len(log_output) - 8
    
    # Original output gap
    x_og = np.column_stack([
        np.ones(t_end + 4),
        np.arange(1, t_end + 5),
        np.concatenate([np.zeros(56), np.arange(1, t_end + 4 - 56 + 1)]) if t_end + 4 > 56 else np.zeros(t_end + 4),
        np.concatenate([np.zeros(142), np.arange(1, t_end + 4 - 142 + 1)]) if t_end + 4 > 142 else np.zeros(t_end + 4),
    ])
    y_og = log_output[4:t_end + 8]
    output_gap = (y_og - x_og @ np.linalg.solve(x_og.T @ x_og, x_og.T @ y_og)) * 100
    
    # IS curve
    y_is = output_gap[4:t_end + 4]
    y_is_l1 = output_gap[3:t_end + 3]
    y_is_l2 = output_gap[2:t_end + 2]
    d = covid_dummy[8:t_end + 8]
    d_l1 = covid_dummy[7:t_end + 7]
    d_l2 = covid_dummy[6:t_end + 6]
    ir_is = (real_interest_rate[7:t_end + 7] + real_interest_rate[6:t_end + 6]) / 2
    
    if sample_end[0] >= 2020 and fix_phi is None:
        print("Stage 3 initial IS: NLS with phi")
        def is_residuals(params):
            phi, a1, a2, ar, a0 = params
            pred = phi * d + a1 * (y_is_l1 - phi * d_l1) + a2 * (y_is_l2 - phi * d_l2) + ar * ir_is + a0
            return y_is - pred
        
        result = least_squares(is_residuals, [0, 0, 0, 0, 0])
        b_is = {"phi": result.x[0], "a_1": result.x[1], "a_2": result.x[2], "a_r": result.x[3], "a_0": result.x[4]}
    else:
        print("Stage 3 initial IS: OLS without phi")
        def is_residuals(params):
            a1, a2, ar, a0 = params
            pred = a1 * y_is_l1 + a2 * y_is_l2 + ar * ir_is + a0
            return y_is - pred
        
        result = least_squares(is_residuals, [0, 0, 0, 0])
        b_is = {"a_1": result.x[0], "a_2": result.x[1], "a_r": result.x[2], "a_0": result.x[3], "phi": 0.0}
    
    if fix_phi is not None:
        b_is["phi"] = fix_phi
    
    r_is = y_is - (b_is["phi"] * d + b_is["a_1"] * (y_is_l1 - b_is["phi"] * d_l1) + 
                   b_is["a_2"] * (y_is_l2 - b_is["phi"] * d_l2) + b_is["a_r"] * ir_is + b_is["a_0"])
    s_is = np.sqrt(np.sum(r_is ** 2) / (len(r_is) - len(b_is)))
    
    # Phillips curve
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
    
    # Initial parameters
    initial_parameters = np.array([
        b_is["a_1"], b_is["a_2"], b_is["a_r"],
        b_ph[0], b_ph[1], b_ph[3], b_ph[4], b_ph[5],
        1.0,  # c
        s_is, s_ph, 0.7, b_is["phi"]
    ])
    
    param_num = {
        "a_1": 1, "a_2": 2, "a_3": 3,
        "b_1": 4, "b_2": 5, "b_3": 6, "b_4": 7, "b_5": 8,
        "c": 9, "sigma_1": 10, "sigma_2": 11, "sigma_4": 12, "phi": 13
    }
    
    n_params = len(initial_parameters)
    
    # Initialize xi_00 from HP filter
    if xi_00 is None:
        print("Stage 3: xi.00 from HP trend in log output")
        g_pot, _ = hpfilter(y_og, lamb=36000)
        g_pot_diff = np.diff(g_pot)
        xi_00 = np.array([
            100 * g_pot[3], 100 * g_pot[2], 100 * g_pot[1],
            100 * g_pot_diff[2], 100 * g_pot_diff[1], 100 * g_pot_diff[0],
            0, 0, 0  # z and lags
        ])
    else:
        print("Stage 3: Using xi.00 input")
    
    # Build data matrices
    y_data = np.column_stack([
        100 * log_output[8:t_end + 8],
        inflation[8:t_end + 8]
    ])
    x_data = np.column_stack([
        100 * log_output[7:t_end + 7],
        100 * log_output[6:t_end + 6],
        real_interest_rate[7:t_end + 7],
        real_interest_rate[6:t_end + 6],
        inflation[7:t_end + 7],
        (inflation[6:t_end + 6] + inflation[5:t_end + 5] + inflation[4:t_end + 4]) / 3,
        (inflation[3:t_end + 3] + inflation[2:t_end + 2] + inflation[1:t_end + 1] + inflation[0:t_end]) / 4,
        relative_oil_price_inflation[7:t_end + 7],
        relative_import_price_inflation[8:t_end + 8],
        covid_dummy[8:t_end + 8],
        covid_dummy[7:t_end + 7],
        covid_dummy[6:t_end + 6],
    ])
    
    # Set bounds
    theta_lb = np.full(n_params, -np.inf)
    theta_ub = np.full(n_params, np.inf)
    
    if b_y_constraint is not None:
        print(f"Setting a lower bound of b_y > {b_y_constraint} in Stage 3")
        if initial_parameters[param_num["b_3"] - 1] < b_y_constraint:
            initial_parameters[param_num["b_3"] - 1] = b_y_constraint
        theta_lb[param_num["b_3"] - 1] = b_y_constraint
    
    if a_r_constraint is not None:
        print(f"Setting an upper bound of a_r < {a_r_constraint} in Stage 3")
        if initial_parameters[param_num["a_3"] - 1] > a_r_constraint:
            initial_parameters[param_num["a_3"] - 1] = a_r_constraint
        theta_ub[param_num["a_3"] - 1] = a_r_constraint
    
    if fix_phi is not None:
        print(f"Fixing phi at {fix_phi}")
        theta_lb[param_num["phi"] - 1] = fix_phi
        theta_ub[param_num["phi"] - 1] = fix_phi
    
    # Add kappa parameters
    if use_kappa and kappa_inputs is not None:
        for k, row in kappa_inputs.iterrows():
            theta_ind = n_params + k
            kappa_inputs.loc[k, "theta.index"] = theta_ind + 1
            param_num[row["name"]] = theta_ind + 1
            initial_parameters = np.append(initial_parameters, row["init"])
            theta_lb = np.append(theta_lb, row["lower.bound"])
            theta_ub = np.append(theta_ub, row["upper.bound"])
            print(f"Initializing {row['name']} at {row['init']}")
    
    # Initialize P_00
    if P_00 is None:
        print("Stage 3: Initializing covariance matrix")
        P_00 = calculate_covariance(
            initial_parameters, theta_lb, theta_ub,
            y_data, x_data, stage, lambda_g, lambda_z, xi_00,
            use_kappa, kappa_inputs, param_num
        )
    else:
        print("Stage 3: Using P.00 input")
    
    # Optimize
    def neg_ll(theta):
        return -_log_likelihood_wrapper(
            theta, y_data, x_data, stage, lambda_g, lambda_z,
            xi_00, P_00, use_kappa, kappa_inputs, param_num
        )
    
    bounds = list(zip(theta_lb, theta_ub))
    result = minimize(
        neg_ll, initial_parameters, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 5000, "ftol": 1e-8}
    )
    
    if not result.success:
        print(f"Warning: Stage 3 optimization: {result.message}")
    else:
        print(f"Stage 3: The terminal conditions in nloptr are {result.message}")
    
    theta = result.x
    log_likelihood = -result.fun
    
    # Get states
    states, matrices = _kalman_states_wrapper(
        theta, y_data, x_data, stage, lambda_g, lambda_z,
        xi_00, P_00, use_kappa, kappa_inputs, param_num
    )
    
    # Extract estimates
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
    
    return Stage3Result(
        theta=theta,
        log_likelihood=log_likelihood,
        lambda_g=lambda_g,
        lambda_z=lambda_z,
        states=states,
        matrices=matrices,
        xi_00=xi_00,
        P_00=P_00,
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
    )
