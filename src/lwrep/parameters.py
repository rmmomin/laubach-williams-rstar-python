"""
State-space matrix builders - direct port of unpack.parameters.stage*.R
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


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
    parameters: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    xi_00: np.ndarray,
    P_00: np.ndarray,
    use_kappa: bool,
    kappa_inputs: Optional[pd.DataFrame],
    param_num: Dict[str, int],
) -> StateSpaceMatrices:
    """Direct port of unpack.parameters.stage1.R"""
    n_state_vars = 3
    
    # A matrix: 10 x 2 (then transposed to match R's t(A))
    A = np.zeros((10, 2))
    A[0, 0] = parameters[param_num["a_1"] - 1]  # a_y,1
    A[1, 0] = parameters[param_num["a_2"] - 1]  # a_y,2
    A[0, 1] = parameters[param_num["b_3"] - 1]  # b_y
    A[2, 1] = parameters[param_num["b_1"] - 1]  # b_{pi,1}
    A[3, 1] = parameters[param_num["b_2"] - 1]  # b_{pi,2-4}
    A[4, 1] = 1 - A[2, 1] - A[3, 1]  # 1 - b_{pi,1} - b_{pi,2-4}
    A[5, 1] = parameters[param_num["b_4"] - 1]  # b_oil
    A[6, 1] = parameters[param_num["b_5"] - 1]  # b_import
    A[7, 0] = parameters[param_num["phi"] - 1]  # phi
    A[8, 0] = -parameters[param_num["a_1"] - 1] * parameters[param_num["phi"] - 1]  # -a_y,1*phi
    A[8, 1] = -parameters[param_num["b_3"] - 1] * parameters[param_num["phi"] - 1]  # -b_y*phi
    A[9, 0] = -parameters[param_num["a_2"] - 1] * parameters[param_num["phi"] - 1]  # -a_y,2*phi
    # Note: R code does NOT transpose A for stage 1 (A stays 10x2)
    
    # H matrix: 3 x 2
    H = np.zeros((3, 2))
    H[0, 0] = 1
    H[1, 0] = -parameters[param_num["a_1"] - 1]  # -a_y,1
    H[2, 0] = -parameters[param_num["a_2"] - 1]  # -a_y,2
    H[1, 1] = -parameters[param_num["b_3"] - 1]  # -b_y
    
    # R matrix
    R = np.diag([
        parameters[param_num["sigma_1"] - 1] ** 2,
        parameters[param_num["sigma_2"] - 1] ** 2
    ])
    
    # Q matrix
    Q = np.zeros((3, 3))
    Q[0, 0] = parameters[param_num["sigma_4"] - 1] ** 2
    
    # F matrix
    F = np.zeros((3, 3))
    F[0, 0] = 1
    F[1, 0] = 1
    F[2, 1] = 1
    
    # Kappa vector
    kappa_vec = np.ones(y_data.shape[0])
    if use_kappa and kappa_inputs is not None:
        for idx, row in kappa_inputs.iterrows():
            T_start = int(row["T.start"]) - 1  # Convert to 0-indexed
            T_end = int(row["T.end"])
            theta_idx = int(row["theta.index"]) - 1  # Convert to 0-indexed
            kappa_vec[T_start:T_end] = parameters[theta_idx]
    
    # cons vector
    cons = np.zeros((n_state_vars, 1))
    cons[0, 0] = parameters[param_num["g"] - 1]
    
    return StateSpaceMatrices(
        xi_00=xi_00,
        P_00=P_00,
        F=F,
        Q=Q,
        A=A,
        H=H,
        R=R,
        kappa_vec=kappa_vec,
        cons=cons,
    )


def unpack_parameters_stage2(
    parameters: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    lambda_g: float,
    xi_00: np.ndarray,
    P_00: np.ndarray,
    use_kappa: bool,
    kappa_inputs: Optional[pd.DataFrame],
    param_num: Dict[str, int],
) -> StateSpaceMatrices:
    """Direct port of unpack.parameters.stage2.R"""
    n_state_vars = 6
    
    # A matrix: 2 x 13 then transposed
    A = np.zeros((2, 13))
    A[0, 0] = parameters[param_num["a_1"] - 1]  # a_y,1
    A[0, 1] = parameters[param_num["a_2"] - 1]  # a_y,2
    A[0, 2] = parameters[param_num["a_3"] - 1] / 2  # a_r/2
    A[0, 3] = parameters[param_num["a_3"] - 1] / 2  # a_r/2
    A[0, 9] = parameters[param_num["a_4"] - 1]  # a_0
    A[1, 0] = parameters[param_num["b_3"] - 1]  # b_y
    A[1, 4] = parameters[param_num["b_1"] - 1]  # b_{pi,1}
    A[1, 5] = parameters[param_num["b_2"] - 1]  # b_{pi,2-4}
    A[1, 6] = 1 - parameters[param_num["b_1"] - 1] - parameters[param_num["b_2"] - 1]
    A[1, 7] = parameters[param_num["b_4"] - 1]  # b_oil
    A[1, 8] = parameters[param_num["b_5"] - 1]  # b_import
    A[0, 10] = parameters[param_num["phi"] - 1]  # phi
    A[0, 11] = -parameters[param_num["a_1"] - 1] * parameters[param_num["phi"] - 1]  # -a_1*phi
    A[0, 12] = -parameters[param_num["a_2"] - 1] * parameters[param_num["phi"] - 1]  # -a_2*phi
    A[1, 11] = -parameters[param_num["b_3"] - 1] * parameters[param_num["phi"] - 1]  # -b_3*phi
    A = A.T  # Transpose to 13 x 2
    
    # H matrix: 2 x 6 then transposed
    H = np.zeros((2, 6))
    H[0, 0] = 1
    H[0, 1] = -parameters[param_num["a_1"] - 1]
    H[0, 2] = -parameters[param_num["a_2"] - 1]
    H[0, 4] = parameters[param_num["a_5"] - 1] / 2  # a_g/2
    H[0, 5] = parameters[param_num["a_5"] - 1] / 2  # a_g/2
    H[1, 1] = -parameters[param_num["b_3"] - 1]
    H = H.T  # Transpose to 6 x 2
    
    # R matrix
    R = np.diag([
        parameters[param_num["sigma_1"] - 1] ** 2,
        parameters[param_num["sigma_2"] - 1] ** 2
    ])
    
    # Q matrix
    Q = np.zeros((6, 6))
    Q[0, 0] = parameters[param_num["sigma_4"] - 1] ** 2
    Q[3, 3] = (lambda_g * parameters[param_num["sigma_4"] - 1]) ** 2
    
    # F matrix
    F = np.zeros((6, 6))
    F[0, 0] = 1
    F[0, 3] = 1
    F[1, 0] = 1
    F[2, 1] = 1
    F[3, 3] = 1
    F[4, 3] = 1
    F[5, 4] = 1
    
    # Kappa vector
    kappa_vec = np.ones(y_data.shape[0])
    if use_kappa and kappa_inputs is not None:
        for idx, row in kappa_inputs.iterrows():
            T_start = int(row["T.start"]) - 1
            T_end = int(row["T.end"])
            theta_idx = int(row["theta.index"]) - 1
            kappa_vec[T_start:T_end] = parameters[theta_idx]
    
    cons = np.zeros((n_state_vars, 1))
    
    return StateSpaceMatrices(
        xi_00=xi_00,
        P_00=P_00,
        F=F,
        Q=Q,
        A=A,
        H=H,
        R=R,
        kappa_vec=kappa_vec,
        cons=cons,
    )


def unpack_parameters_stage3(
    parameters: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    lambda_g: float,
    lambda_z: float,
    xi_00: np.ndarray,
    P_00: np.ndarray,
    use_kappa: bool,
    kappa_inputs: Optional[pd.DataFrame],
    param_num: Dict[str, int],
) -> StateSpaceMatrices:
    """Direct port of unpack.parameters.stage3.R"""
    n_state_vars = 9
    
    # A matrix: 2 x 12 then transposed
    A = np.zeros((2, 12))
    A[0, 0] = parameters[param_num["a_1"] - 1]  # a_y,1
    A[0, 1] = parameters[param_num["a_2"] - 1]  # a_y,2
    A[0, 2] = parameters[param_num["a_3"] - 1] / 2  # a_r/2
    A[0, 3] = parameters[param_num["a_3"] - 1] / 2  # a_r/2
    A[1, 0] = parameters[param_num["b_3"] - 1]  # b_y
    A[1, 4] = parameters[param_num["b_1"] - 1]  # b_{pi,1}
    A[1, 5] = parameters[param_num["b_2"] - 1]  # b_{pi,2-4}
    A[1, 6] = 1 - parameters[param_num["b_1"] - 1] - parameters[param_num["b_2"] - 1]
    A[1, 7] = parameters[param_num["b_4"] - 1]  # b_oil
    A[1, 8] = parameters[param_num["b_5"] - 1]  # b_import
    A[0, 9] = parameters[param_num["phi"] - 1]  # phi
    A[0, 10] = -parameters[param_num["a_1"] - 1] * parameters[param_num["phi"] - 1]  # -a_y,1*phi
    A[0, 11] = -parameters[param_num["a_2"] - 1] * parameters[param_num["phi"] - 1]  # -a_y,2*phi
    A[1, 10] = -parameters[param_num["b_3"] - 1] * parameters[param_num["phi"] - 1]  # -b_y*phi
    A = A.T  # Transpose to 12 x 2
    
    # H matrix: 2 x 9 then transposed
    H = np.zeros((2, 9))
    H[0, 0] = 1
    H[0, 1] = -parameters[param_num["a_1"] - 1]
    H[0, 2] = -parameters[param_num["a_2"] - 1]
    H[0, 4] = -parameters[param_num["c"] - 1] * parameters[param_num["a_3"] - 1] * 2  # -c*a_r (annualized)
    H[0, 5] = -parameters[param_num["c"] - 1] * parameters[param_num["a_3"] - 1] * 2
    H[0, 7] = -parameters[param_num["a_3"] - 1] / 2  # -a_r/2
    H[0, 8] = -parameters[param_num["a_3"] - 1] / 2
    H[1, 1] = -parameters[param_num["b_3"] - 1]
    H = H.T  # Transpose to 9 x 2
    
    # R matrix
    R = np.diag([
        parameters[param_num["sigma_1"] - 1] ** 2,
        parameters[param_num["sigma_2"] - 1] ** 2
    ])
    
    # Q matrix
    Q = np.zeros((9, 9))
    Q[0, 0] = parameters[param_num["sigma_4"] - 1] ** 2
    Q[3, 3] = (lambda_g * parameters[param_num["sigma_4"] - 1]) ** 2
    Q[6, 6] = (lambda_z * parameters[param_num["sigma_1"] - 1] / parameters[param_num["a_3"] - 1]) ** 2
    
    # F matrix
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
    
    # Kappa vector
    kappa_vec = np.ones(y_data.shape[0])
    if use_kappa and kappa_inputs is not None:
        for idx, row in kappa_inputs.iterrows():
            T_start = int(row["T.start"]) - 1
            T_end = int(row["T.end"])
            theta_idx = int(row["theta.index"]) - 1
            kappa_vec[T_start:T_end] = parameters[theta_idx]
    
    cons = np.zeros((n_state_vars, 1))
    
    return StateSpaceMatrices(
        xi_00=xi_00,
        P_00=P_00,
        F=F,
        Q=Q,
        A=A,
        H=H,
        R=R,
        kappa_vec=kappa_vec,
        cons=cons,
    )
