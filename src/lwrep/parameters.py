from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


def _matrix_power(mat: np.ndarray, power: int) -> np.ndarray:
    if power == 0:
        return np.eye(mat.shape[0])
    result = np.eye(mat.shape[0])
    base = mat.copy()
    p = power
    while p > 0:
        if p % 2 == 1:
            result = result @ base
        base = base @ base
        p //= 2
    return result


def _initialize_stage1(
    A: np.ndarray,
    H: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    cons: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Ht = H.T
    F1 = F
    F2 = F @ F
    F3 = F2 @ F
    x_stack = np.vstack([Ht, Ht @ F1, Ht @ F2, Ht @ F3])
    om = np.zeros((8, 8))

    om[np.ix_(range(4, 6), range(2, 4))] = Ht @ F1 @ Q @ H
    om[np.ix_(range(6, 8), range(2, 4))] = Ht @ F2 @ Q @ H
    om[np.ix_(range(6, 8), range(4, 6))] = Ht @ (F2 @ Q @ F1.T + F1 @ Q) @ H

    om = om + om.T
    om[np.ix_(range(0, 2), range(0, 2))] = R
    om[np.ix_(range(2, 4), range(2, 4))] = Ht @ Q @ H + R
    om[np.ix_(range(4, 6), range(4, 6))] = Ht @ (F1 @ Q @ F1.T + Q) @ H + R
    om[np.ix_(range(6, 8), range(6, 8))] = Ht @ (F2 @ Q @ (F1.T @ F1.T) + F1 @ Q @ F1.T + Q) @ H + R

    p1 = x_stack.T @ np.linalg.solve(om, x_stack)
    yy = np.concatenate([y_data[i] for i in range(4)])

    tmp = np.concatenate(
        [
            A.T @ x_data[0],
            A.T @ x_data[1] + Ht @ cons,
            A.T @ x_data[2] + Ht @ cons + Ht @ F1 @ cons,
            A.T @ x_data[3] + Ht @ (np.eye(F.shape[0]) + F1 + F2) @ cons,
        ]
    )
    diff = yy - tmp
    xi0 = np.linalg.solve(p1, x_stack.T @ np.linalg.solve(om, diff))
    resid = diff - x_stack @ xi0
    scale = np.sum(resid**2) / 3.0
    P0 = np.linalg.solve(p1, np.eye(3) * scale)
    return xi0, P0


def build_stage1_system(
    theta: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    xi0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> StateSpace:
    A = np.zeros((7, 2))
    A[0:2, 0] = theta[0:2]
    A[0, 1] = theta[4]
    A[2:4, 1] = theta[2:4]
    A[4, 1] = 1.0 - np.sum(A[2:4, 1])
    A[5:7, 1] = theta[5:7]

    H = np.zeros((3, 2))
    H[0, 0] = 1.0
    H[1:3, 0] = -theta[0:2]
    H[1, 1] = -theta[4]

    R = np.diag([theta[8] ** 2, theta[9] ** 2])
    Q = np.zeros((3, 3))
    Q[0, 0] = theta[10] ** 2

    F = np.zeros((3, 3))
    F[0, 0] = F[1, 0] = F[2, 1] = 1.0

    cons = np.zeros(3)
    cons[0] = theta[7]

    if xi0 is None or P0 is None:
        xi0, P0 = _initialize_stage1(A, H, F, Q, R, cons, y_data, x_data)

    return StateSpace(xi0=xi0, P0=P0, F=F, Q=Q, A=A, H=H, R=R, cons=cons)


def _initialize_stage2(
    A: np.ndarray,
    H: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Ht = H.T
    F1 = F
    F2 = F @ F
    F3 = F2 @ F
    F4 = F3 @ F

    x_stack = np.vstack([Ht, Ht @ F1, Ht @ F2, Ht @ F3, Ht @ F4])
    om = np.zeros((10, 10))

    om[np.ix_(range(4, 6), range(2, 4))] = Ht @ F1 @ Q @ H
    om[np.ix_(range(6, 8), range(2, 4))] = Ht @ F2 @ Q @ H
    om[np.ix_(range(8, 10), range(2, 4))] = Ht @ F3 @ Q @ H
    om[np.ix_(range(6, 8), range(4, 6))] = Ht @ (F2 @ Q @ F1.T + F1 @ Q) @ H
    om[np.ix_(range(8, 10), range(4, 6))] = Ht @ F1 @ (F2 @ Q @ F1.T + F1 @ Q) @ H
    om[np.ix_(range(8, 10), range(6, 8))] = Ht @ F1 @ (F2 @ Q @ (F1.T @ F1.T) + F1 @ Q @ F1.T + Q) @ H

    om = om + om.T
    om[np.ix_(range(0, 2), range(0, 2))] = R
    om[np.ix_(range(2, 4), range(2, 4))] = Ht @ Q @ H + R
    om[np.ix_(range(4, 6), range(4, 6))] = Ht @ (F1 @ Q @ F1.T + Q) @ H + R
    om[np.ix_(range(6, 8), range(6, 8))] = Ht @ (F2 @ Q @ (F1.T @ F1.T) + F1 @ Q @ F1.T + Q) @ H + R
    om[np.ix_(range(8, 10), range(8, 10))] = Ht @ (
        F3 @ Q @ (F1.T @ F1.T @ F1.T) + F2 @ Q @ (F1.T @ F1.T) + F1 @ Q @ F1.T + Q
    ) @ H + R

    p1 = x_stack.T @ np.linalg.solve(om, x_stack)
    yy = np.concatenate([y_data[i] for i in range(5)])
    tmp = np.concatenate([A.T @ x_data[i] for i in range(5)])

    diff = yy - tmp
    xi0 = np.linalg.solve(p1, x_stack.T @ np.linalg.solve(om, diff))
    resid = diff - x_stack @ xi0
    scale = np.sum(resid**2) / (len(yy) - F.shape[0])
    P0 = np.linalg.solve(p1, np.eye(F.shape[0]) * scale)
    return xi0, P0


def build_stage2_system(
    theta: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    lambda_g: float,
    xi0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
) -> StateSpace:
    A = np.zeros((2, 10))
    A[0, 0:2] = theta[0:2]
    A[0, 2:4] = theta[2] / 2.0
    A[0, 9] = theta[3]
    A[1, 0] = theta[7]
    A[1, 4:6] = theta[5:7]
    A[1, 6] = 1.0 - theta[5] - theta[6]
    A[1, 7:9] = theta[8:10]
    A = A.T

    H = np.zeros((2, 4))
    H[0, 0] = 1.0
    H[0, 1:3] = -theta[0:2]
    H[0, 3] = theta[4]
    H[1, 1] = -theta[7]
    H = H.T

    R = np.diag([theta[10] ** 2, theta[11] ** 2])

    Q = np.zeros((4, 4))
    Q[0, 0] = theta[12] ** 2
    Q[3, 3] = (lambda_g * theta[12]) ** 2

    F = np.zeros((4, 4))
    F[0, 0] = F[0, 3] = F[1, 0] = F[2, 1] = F[3, 3] = 1.0
    cons = np.zeros(4)

    if xi0 is None or P0 is None:
        xi0, P0 = _initialize_stage2(A, H, F, Q, R, y_data, x_data)

    return StateSpace(xi0=xi0, P0=P0, F=F, Q=Q, A=A, H=H, R=R, cons=cons)


def _initialize_stage3(
    A: np.ndarray,
    H: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    xi0_gpot: np.ndarray,
    P0_gpot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    Ht = H.T
    F1 = F
    F2 = F @ F
    F3 = F2 @ F
    F4 = F3 @ F

    x_stack = np.vstack([Ht, Ht @ F1, Ht @ F2, Ht @ F3, Ht @ F4])
    om = np.zeros((10, 10))

    om[np.ix_(range(4, 6), range(2, 4))] = Ht @ F1 @ Q @ H
    om[np.ix_(range(6, 8), range(2, 4))] = Ht @ F2 @ Q @ H
    om[np.ix_(range(8, 10), range(2, 4))] = Ht @ F3 @ Q @ H
    om[np.ix_(range(6, 8), range(4, 6))] = Ht @ F1 @ (F1 @ Q @ F1.T + Q) @ H
    om[np.ix_(range(8, 10), range(4, 6))] = Ht @ F2 @ (F1 @ Q @ F1.T + Q) @ H
    om[np.ix_(range(8, 10), range(6, 8))] = Ht @ F1 @ (F2 @ Q @ (F1.T @ F1.T) + F1 @ Q @ F1.T + Q) @ H

    om = om + om.T
    om[np.ix_(range(0, 2), range(0, 2))] = R
    om[np.ix_(range(2, 4), range(2, 4))] = Ht @ Q @ H + R
    om[np.ix_(range(4, 6), range(4, 6))] = Ht @ (F1 @ Q @ F1.T + Q) @ H + R
    om[np.ix_(range(6, 8), range(6, 8))] = Ht @ (F2 @ Q @ (F1.T @ F1.T) + F1 @ Q @ F1.T + Q) @ H + R
    om[np.ix_(range(8, 10), range(8, 10))] = Ht @ (
        F3 @ Q @ (F1.T @ F1.T @ F1.T) + F2 @ Q @ (F1.T @ F1.T) + F1 @ Q @ F1.T + Q
    ) @ H + R

    det_om = np.linalg.det(om)
    p1 = x_stack.T @ np.linalg.solve(om, x_stack) if det_om > 1e-16 else None
    if det_om <= 1e-16 or np.linalg.det(p1) <= 1e-16:
        return xi0_gpot, P0_gpot, False

    yy = np.concatenate([y_data[i] for i in range(5)])
    tmp = np.concatenate([A.T @ x_data[i] for i in range(5)])
    diff = yy - tmp
    xi0 = np.linalg.solve(p1, x_stack.T @ np.linalg.solve(om, diff))
    resid = diff - x_stack @ xi0
    scale = np.sum(resid**2) / (len(yy) - F.shape[0])
    P0 = np.linalg.solve(p1, np.eye(F.shape[0]) * scale)
    return xi0, P0, True


def build_stage3_system(
    theta: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    lambda_g: float,
    lambda_z: float,
    xi0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    xi0_gpot: Optional[np.ndarray] = None,
    P0_gpot: Optional[np.ndarray] = None,
) -> tuple[StateSpace, bool]:
    if xi0_gpot is None or P0_gpot is None:
        raise ValueError("xi0_gpot and P0_gpot must be provided for stage 3.")

    A = np.zeros((2, 9))
    A[0, 0:2] = theta[0:2]
    A[0, 2:4] = theta[2] / 2.0
    A[1, 0] = theta[5]
    A[1, 4:6] = theta[3:5]
    A[1, 6] = 1.0 - theta[3] - theta[4]
    A[1, 7:9] = theta[6:8]
    A = A.T

    H = np.zeros((2, 7))
    H[0, 0] = 1.0
    H[0, 1:3] = -theta[0:2]
    H[0, 3:5] = -theta[8] * theta[2] * 2.0
    H[0, 5:7] = -theta[2] / 2.0
    H[1, 1] = -theta[5]
    H = H.T

    R = np.diag([theta[9] ** 2, theta[10] ** 2])

    Q = np.zeros((7, 7))
    Q[0, 0] = (1 + lambda_g**2) * theta[11] ** 2
    Q[0, 3] = Q[3, 0] = (lambda_g * theta[11]) ** 2
    Q[3, 3] = (lambda_g * theta[11]) ** 2
    Q[5, 5] = (lambda_z * theta[9] / theta[2]) ** 2

    F = np.zeros((7, 7))
    F[0, 0] = F[0, 3] = F[1, 0] = F[2, 1] = F[3, 3] = F[4, 3] = F[5, 5] = F[6, 5] = 1.0
    cons = np.zeros(7)

    gls = False
    if xi0 is None or P0 is None:
        xi0, P0, gls = _initialize_stage3(
            A, H, F, Q, R, y_data, x_data, xi0_gpot, P0_gpot
        )

    return StateSpace(xi0=xi0, P0=P0, F=F, Q=Q, A=A, H=H, R=R, cons=cons), gls


def calculate_covariance(
    initial_theta: np.ndarray,
    bounds: list[tuple[float, float]],
    y_data: np.ndarray,
    x_data: np.ndarray,
    stage: int,
    lambda_g: Optional[float] = None,
    lambda_z: Optional[float] = None,
    xi0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Python port of calculate.covariance.R.

    Parameters
    ----------
    initial_theta : np.ndarray
        Starting values for the parameters.
    bounds : list of tuple
        Bounds passed to scipy.optimize.minimize.
    stage : int
        1, 2, or 3.
    xi0 : np.ndarray
        Initial state vector used during the covariance search (stage 3).
    """

    if xi0 is None:
        raise ValueError("xi0 must be provided for calculate_covariance.")

    n_state = xi0.size
    P0 = np.eye(n_state) * 0.2

    def objective(theta: np.ndarray) -> float:
        ss = _build_state_space_for_stage(stage, theta, y_data, x_data, lambda_g, lambda_z, xi0, P0)
        from .kalman import kalman_filter

        filtered = kalman_filter(
            ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons, y_data, x_data
        )
        return -filtered.loglik

    res = minimize(objective, initial_theta, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
    if not res.success:
        raise RuntimeError(f"Covariance optimization failed: {res.message}")

    ss = _build_state_space_for_stage(stage, res.x, y_data, x_data, lambda_g, lambda_z, xi0, P0)
    from .kalman import kalman_filter

    filtered = kalman_filter(
        ss.xi0, ss.P0, ss.F, ss.Q, ss.A, ss.H, ss.R, ss.cons, y_data, x_data
    )
    return filtered.cov_pred[0]


def _build_state_space_for_stage(
    stage: int,
    theta: np.ndarray,
    y_data: np.ndarray,
    x_data: np.ndarray,
    lambda_g: Optional[float],
    lambda_z: Optional[float],
    xi0: np.ndarray,
    P0: np.ndarray,
) -> StateSpace:
    if stage == 1:
        return build_stage1_system(theta, y_data, x_data, xi0=xi0, P0=P0)
    if stage == 2:
        if lambda_g is None:
            raise ValueError("lambda_g is required for stage 2.")
        return build_stage2_system(theta, y_data, x_data, lambda_g, xi0=xi0, P0=P0)
    if stage == 3:
        if lambda_g is None or lambda_z is None:
            raise ValueError("lambda_g and lambda_z are required for stage 3.")
        state_space, _ = build_stage3_system(
            theta,
            y_data,
            x_data,
            lambda_g,
            lambda_z,
            xi0=xi0,
            P0=P0,
            xi0_gpot=xi0,
            P0_gpot=P0,
        )
        return state_space
    raise ValueError(f"Unsupported stage: {stage}")


