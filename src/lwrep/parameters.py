"""
State-space parameter builders for LW 2023 replication.

Stage 1: 3 states [y*, y*_{t-1}, y*_{t-2}], 12 base params
Stage 2: 6 states [y*, y*_{t-1}, y*_{t-2}, g, g_{t-1}, g_{t-2}], 14 base params
Stage 3: 9 states [y*, y*_{t-1}, y*_{t-2}, g, g_{t-1}, g_{t-2}, z, z_{t-1}, z_{t-2}], 13 base params
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize


@dataclass
class StateSpace:
    xi0: np.ndarray
    P0: np.ndarray
    F: np.ndarray
    Q: np.ndarray
    A: np.ndarray
    H: np.ndarray
    R: np.ndarray
    cons: np.ndarray
    kappa_vec: np.ndarray


# Parameter index mappings (0-indexed, matching R param.num - 1)
PARAM_NUM_STAGE1 = {
    "a_1": 0, "a_2": 1, "b_1": 2, "b_2": 3, "b_3": 4, "b_4": 5, "b_5": 6,
    "g": 7, "sigma_1": 8, "sigma_2": 9, "sigma_4": 10, "phi": 11,
}

PARAM_NUM_STAGE2 = {
    "a_1": 0, "a_2": 1, "a_3": 2, "a_4": 3, "a_5": 4,
    "b_1": 5, "b_2": 6, "b_3": 7, "b_4": 8, "b_5": 9,
    "sigma_1": 10, "sigma_2": 11, "sigma_4": 12, "phi": 13,
}

PARAM_NUM_STAGE3 = {
    "a_1": 0, "a_2": 1, "a_3": 2,
    "b_1": 3, "b_2": 4, "b_3": 5, "b_4": 6, "b_5": 7,
    "c": 8, "sigma_1": 9, "sigma_2": 10, "sigma_4": 11, "phi": 12,
}


def build_kappa_vector(
    T: int,
    parameters: np.ndarray,
    use_kappa: bool,
    kappa_inputs: list,
    n_base_params: int,
) -> np.ndarray:
    """Build the time-varying variance scale vector."""
    kappa_vec = np.ones(T)
    if use_kappa and kappa_inputs:
        for k, kappa_cfg in enumerate(kappa_inputs):
            theta_idx = n_base_params + k
            if theta_idx < len(parameters):
                T_start = kappa_cfg.T_start
                T_end = min(kappa_cfg.T_end, T - 1)
                if T_start < T and T_end >= T_start:
                    kappa_vec[T_start : T_end + 1] = parameters[theta_idx]
    return kappa_vec


def build_stage1_system(
    parameters: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    xi0: np.ndarray,
    P0: np.ndarray,
    use_kappa: bool = True,
    kappa_inputs: Optional[list] = None,
) -> StateSpace:
    """
    Build stage 1 state-space system matrices.
    Stage 1 has 3 state variables: [y*, y*_{t-1}, y*_{t-2}]
    x_data has 10 columns.
    """
    n_state = 3
    T = y_data.shape[0]
    p = PARAM_NUM_STAGE1
    
    # A matrix: 10 x 2
    A = np.zeros((10, 2))
    A[0, 0] = parameters[p["a_1"]]  # a_y,1
    A[1, 0] = parameters[p["a_2"]]  # a_y,2
    A[0, 1] = parameters[p["b_3"]]  # b_y
    A[2, 1] = parameters[p["b_1"]]  # b_{pi,1}
    A[3, 1] = parameters[p["b_2"]]  # b_{pi,2-4}
    A[4, 1] = 1.0 - A[2, 1] - A[3, 1]
    A[5, 1] = parameters[p["b_4"]]  # b_oil
    A[6, 1] = parameters[p["b_5"]]  # b_import
    A[7, 0] = parameters[p["phi"]]  # phi
    A[8, 0] = -parameters[p["a_1"]] * parameters[p["phi"]]  # -a_1*phi
    A[8, 1] = -parameters[p["b_3"]] * parameters[p["phi"]]  # -b_3*phi
    A[9, 0] = -parameters[p["a_2"]] * parameters[p["phi"]]  # -a_2*phi
    
    # H matrix: 3 x 2
    H = np.zeros((n_state, 2))
    H[0, 0] = 1.0
    H[1, 0] = -parameters[p["a_1"]]
    H[2, 0] = -parameters[p["a_2"]]
    H[1, 1] = -parameters[p["b_3"]]
    
    # R matrix
    R = np.diag([parameters[p["sigma_1"]] ** 2, parameters[p["sigma_2"]] ** 2])
    
    # Q matrix
    Q = np.zeros((n_state, n_state))
    Q[0, 0] = parameters[p["sigma_4"]] ** 2
    
    # F matrix
    F = np.zeros((n_state, n_state))
    F[0, 0] = 1.0
    F[1, 0] = 1.0
    F[2, 1] = 1.0
    
    # cons vector (drift)
    cons = np.zeros(n_state)
    cons[0] = parameters[p["g"]]  # trend growth
    
    kappa_vec = build_kappa_vector(T, parameters, use_kappa, kappa_inputs or [], n_base_params=12)
    
    return StateSpace(xi0=xi0, P0=P0, F=F, Q=Q, A=A, H=H, R=R, cons=cons, kappa_vec=kappa_vec)


def build_stage2_system(
    parameters: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    lambda_g: float,
    xi0: np.ndarray,
    P0: np.ndarray,
    use_kappa: bool = True,
    kappa_inputs: Optional[list] = None,
) -> StateSpace:
    """
    Build stage 2 state-space system matrices.
    Stage 2 has 6 state variables: [y*, y*_{t-1}, y*_{t-2}, g, g_{t-1}, g_{t-2}]
    x_data has 13 columns.
    """
    n_state = 6
    T = y_data.shape[0]
    p = PARAM_NUM_STAGE2
    
    # A matrix: 13 x 2, will be transposed
    A = np.zeros((2, 13))
    A[0, 0] = parameters[p["a_1"]]  # a_y,1
    A[0, 1] = parameters[p["a_2"]]  # a_y,2
    A[0, 2:4] = parameters[p["a_3"]] / 2.0  # a_r/2
    A[0, 9] = parameters[p["a_4"]]  # a_0
    A[1, 0] = parameters[p["b_3"]]  # b_y
    A[1, 4] = parameters[p["b_1"]]  # b_{pi,1}
    A[1, 5] = parameters[p["b_2"]]  # b_{pi,2-4}
    A[1, 6] = 1.0 - parameters[p["b_1"]] - parameters[p["b_2"]]
    A[1, 7] = parameters[p["b_4"]]  # b_oil
    A[1, 8] = parameters[p["b_5"]]  # b_import
    A[0, 10] = parameters[p["phi"]]  # phi
    A[0, 11] = -parameters[p["a_1"]] * parameters[p["phi"]]  # -a_1*phi
    A[0, 12] = -parameters[p["a_2"]] * parameters[p["phi"]]  # -a_2*phi
    A[1, 11] = -parameters[p["b_3"]] * parameters[p["phi"]]  # -b_3*phi
    A = A.T  # 13 x 2
    
    # H matrix: 6 x 2, will be transposed
    H = np.zeros((2, n_state))
    H[0, 0] = 1.0
    H[0, 1] = -parameters[p["a_1"]]
    H[0, 2] = -parameters[p["a_2"]]
    H[0, 4:6] = parameters[p["a_5"]] / 2.0  # a_g/2 for (g_{t-1} + g_{t-2})
    H[1, 1] = -parameters[p["b_3"]]
    H = H.T  # 6 x 2
    
    # R matrix
    R = np.diag([parameters[p["sigma_1"]] ** 2, parameters[p["sigma_2"]] ** 2])
    
    # Q matrix
    Q = np.zeros((n_state, n_state))
    Q[0, 0] = parameters[p["sigma_4"]] ** 2
    Q[3, 3] = (lambda_g * parameters[p["sigma_4"]]) ** 2
    
    # F matrix
    F = np.zeros((n_state, n_state))
    F[0, 0] = F[0, 3] = F[1, 0] = F[2, 1] = F[3, 3] = F[4, 3] = F[5, 4] = 1.0
    
    cons = np.zeros(n_state)
    kappa_vec = build_kappa_vector(T, parameters, use_kappa, kappa_inputs or [], n_base_params=14)
    
    return StateSpace(xi0=xi0, P0=P0, F=F, Q=Q, A=A, H=H, R=R, cons=cons, kappa_vec=kappa_vec)


def build_stage3_system(
    parameters: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    lambda_g: float,
    lambda_z: float,
    xi0: np.ndarray,
    P0: np.ndarray,
    use_kappa: bool = True,
    kappa_inputs: Optional[list] = None,
) -> StateSpace:
    """
    Build stage 3 state-space system matrices.
    Stage 3 has 9 state variables: [y*, y*_{t-1}, y*_{t-2}, g, g_{t-1}, g_{t-2}, z, z_{t-1}, z_{t-2}]
    x_data has 12 columns.
    """
    n_state = 9
    T = y_data.shape[0]
    p = PARAM_NUM_STAGE3
    
    # A matrix: 12 x 2, will be transposed
    A = np.zeros((2, 12))
    A[0, 0] = parameters[p["a_1"]]  # a_y,1
    A[0, 1] = parameters[p["a_2"]]  # a_y,2
    A[0, 2:4] = parameters[p["a_3"]] / 2.0  # a_r/2
    A[1, 0] = parameters[p["b_3"]]  # b_y
    A[1, 4] = parameters[p["b_1"]]  # b_{pi,1}
    A[1, 5] = parameters[p["b_2"]]  # b_{pi,2-4}
    A[1, 6] = 1.0 - parameters[p["b_1"]] - parameters[p["b_2"]]
    A[1, 7] = parameters[p["b_4"]]  # b_oil
    A[1, 8] = parameters[p["b_5"]]  # b_import
    A[0, 9] = parameters[p["phi"]]  # phi
    A[0, 10] = -parameters[p["a_1"]] * parameters[p["phi"]]  # -a_y,1*phi
    A[0, 11] = -parameters[p["a_2"]] * parameters[p["phi"]]  # -a_y,2*phi
    A[1, 10] = -parameters[p["b_3"]] * parameters[p["phi"]]  # -b_y*phi
    A = A.T  # 12 x 2
    
    # H matrix: 9 x 2, will be transposed
    H = np.zeros((2, n_state))
    H[0, 0] = 1.0
    H[0, 1] = -parameters[p["a_1"]]
    H[0, 2] = -parameters[p["a_2"]]
    H[0, 4:6] = -parameters[p["c"]] * parameters[p["a_3"]] * 2.0  # c * a_r annualized
    H[0, 7:9] = -parameters[p["a_3"]] / 2.0  # a_r/2 for z lags
    H[1, 1] = -parameters[p["b_3"]]
    H = H.T  # 9 x 2
    
    # R matrix
    R = np.diag([parameters[p["sigma_1"]] ** 2, parameters[p["sigma_2"]] ** 2])
    
    # Q matrix
    Q = np.zeros((n_state, n_state))
    Q[0, 0] = parameters[p["sigma_4"]] ** 2  # sigma_y*
    Q[3, 3] = (lambda_g * parameters[p["sigma_4"]]) ** 2  # sigma_g
    Q[6, 6] = (lambda_z * parameters[p["sigma_1"]] / parameters[p["a_3"]]) ** 2  # sigma_z
    
    # F matrix
    F = np.zeros((n_state, n_state))
    F[0, 0] = F[0, 3] = 1.0  # y* = y* + g
    F[1, 0] = 1.0  # y*_{t-1} = y*
    F[2, 1] = 1.0  # y*_{t-2} = y*_{t-1}
    F[3, 3] = 1.0  # g = g (random walk)
    F[4, 3] = 1.0  # g_{t-1} = g
    F[5, 4] = 1.0  # g_{t-2} = g_{t-1}
    F[6, 6] = 1.0  # z = z (random walk)
    F[7, 6] = 1.0  # z_{t-1} = z
    F[8, 7] = 1.0  # z_{t-2} = z_{t-1}
    
    cons = np.zeros(n_state)
    kappa_vec = build_kappa_vector(T, parameters, use_kappa, kappa_inputs or [], n_base_params=13)
    
    return StateSpace(xi0=xi0, P0=P0, F=F, Q=Q, A=A, H=H, R=R, cons=cons, kappa_vec=kappa_vec)


def calculate_initial_covariance(
    initial_parameters: np.ndarray,
    bounds: List[tuple],
    y_data: np.ndarray,
    x_data: np.ndarray,
    stage: int,
    lambda_g: float,
    lambda_z: float,
    xi0: np.ndarray,
    use_kappa: bool = True,
    kappa_inputs: Optional[list] = None,
) -> np.ndarray:
    """Calculate initial covariance matrix P0 via optimization."""
    n_state = xi0.size
    P0_init = np.eye(n_state) * 0.2
    
    from .kalman import kalman_filter
    
    def objective(theta: np.ndarray) -> float:
        try:
            if stage == 3:
                ss = build_stage3_system(
                    theta, y_data, x_data, lambda_g, lambda_z, xi0, P0_init,
                    use_kappa, kappa_inputs
                )
            elif stage == 2:
                ss = build_stage2_system(
                    theta, y_data, x_data, lambda_g, xi0, P0_init,
                    use_kappa, kappa_inputs
                )
            else:
                ss = build_stage1_system(
                    theta, y_data, x_data, xi0, P0_init,
                    use_kappa, kappa_inputs
                )
            filtered = kalman_filter(
                ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons,
                y_data, x_data, ss.kappa_vec
            )
            return -filtered.loglik
        except (np.linalg.LinAlgError, ValueError):
            return 1e10
    
    res = minimize(
        objective, initial_parameters, method="L-BFGS-B",
        bounds=bounds, options={"maxiter": 1000}
    )
    
    # Get final covariance from filter
    if stage == 3:
        ss = build_stage3_system(
            res.x, y_data, x_data, lambda_g, lambda_z, xi0, P0_init,
            use_kappa, kappa_inputs
        )
    elif stage == 2:
        ss = build_stage2_system(
            res.x, y_data, x_data, lambda_g, xi0, P0_init,
            use_kappa, kappa_inputs
        )
    else:
        ss = build_stage1_system(
            res.x, y_data, x_data, xi0, P0_init,
            use_kappa, kappa_inputs
        )
    
    filtered = kalman_filter(
        ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons,
        y_data, x_data, ss.kappa_vec
    )
    return filtered.cov_pred[0]
