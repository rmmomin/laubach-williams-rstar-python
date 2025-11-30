#!/usr/bin/env python3
"""
Bayesian Laubach-Williams-style r* estimator (no hours).

States (7):
  0: y_gap_t      = 100 * (y_t - y*_t)      # output gap in %
  1: y_gap_{t-1}
  2: pi_t         # inflation
  3: pi_{t-1}
  4: y_star_t     # log potential output
  5: g_t          # trend growth (quarterly, ann.)
  6: z_t          # other component of r*

Transition:
  y_gap_t = phi1 * y_gap_{t-1}
            + gamma * (r_{t-1} - (c * g_{t-1} + z_{t-1})) + eps_y

  y_gap_{t-1} <- y_gap_t (lag)
  pi_t        = alpha * pi_{t-1} + kappa * y_gap_{t-1} + eps_pi
  pi_{t-1}    <- pi_t (lag)

  y_star_t    = y_star_{t-1} + g_{t-1} + eps_y_star
  g_t         = g_{t-1} + eps_g
  z_t         = z_{t-1} + eps_z

Measurement:
  y_obs_t  = y_star_t + y_gap_t / 100 + meas_y_noise
  pi_obs_t = pi_t + meas_pi_noise

We integrate out the states analytically via a Kalman filter
and do Bayesian inference on parameters via PyMC, using
literature-inspired priors to avoid the pile-up-at-zero problem.
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------
# 1. Kalman filter and smoother
# ---------------------------------------------------------------------

@dataclass
class KFResults:
    loglik: float
    filtered_state: np.ndarray
    filtered_cov: np.ndarray
    pred_state: np.ndarray
    pred_cov: np.ndarray


def kalman_filter(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    c_t: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> KFResults:
    """
    Linear Gaussian Kalman filter with time-varying state intercept c_t.
    """
    T, n_obs = y.shape
    n_states = F.shape[0]

    filtered_state = np.zeros((T, n_states))
    filtered_cov = np.zeros((T, n_states, n_states))
    pred_state = np.zeros((T, n_states))
    pred_cov = np.zeros((T, n_states, n_states))

    x_filt = x0.copy()
    P_filt = P0.copy()
    loglik = 0.0
    two_pi = 2.0 * np.pi

    for t in range(T):
        # Predict
        x_pred = F @ x_filt + c_t[t]
        P_pred = F @ P_filt @ F.T + Q

        # Forecast y
        y_pred = H @ x_pred
        v = y[t] - y_pred
        S = H @ P_pred @ H.T + R

        try:
            sign, logdet = np.linalg.slogdet(S)
        except np.linalg.LinAlgError:
            return KFResults(-1e10, filtered_state, filtered_cov, pred_state, pred_cov)

        if sign <= 0:
            return KFResults(-1e10, filtered_state, filtered_cov, pred_state, pred_cov)

        S_inv = np.linalg.inv(S)
        ll_t = -0.5 * (n_obs * np.log(two_pi) + logdet + v.T @ S_inv @ v)
        loglik += ll_t

        # Update
        K = P_pred @ H.T @ S_inv
        x_filt = x_pred + K @ v
        P_filt = (np.eye(n_states) - K @ H) @ P_pred

        filtered_state[t] = x_filt
        filtered_cov[t] = P_filt
        pred_state[t] = x_pred
        pred_cov[t] = P_pred

    return KFResults(
        loglik=loglik,
        filtered_state=filtered_state,
        filtered_cov=filtered_cov,
        pred_state=pred_state,
        pred_cov=pred_cov,
    )


def kalman_smoother(F: np.ndarray, kf_res: KFResults) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel smoother.
    """
    filtered_state = kf_res.filtered_state
    filtered_cov = kf_res.filtered_cov
    pred_state = kf_res.pred_state
    pred_cov = kf_res.pred_cov

    T, n_states = filtered_state.shape
    x_smooth = np.zeros_like(filtered_state)
    P_smooth = np.zeros_like(filtered_cov)

    # Start from last
    x_smooth[-1] = filtered_state[-1]
    P_smooth[-1] = filtered_cov[-1]

    for t in range(T - 2, -1, -1):
        P_filt = filtered_cov[t]
        P_pred_next = pred_cov[t + 1]
        C = P_filt @ F.T @ np.linalg.inv(P_pred_next)
        x_smooth[t] = filtered_state[t] + C @ (x_smooth[t + 1] - pred_state[t + 1])
        P_smooth[t] = filtered_cov[t] + C @ (P_smooth[t + 1] - P_pred_next) @ C.T

    return x_smooth, P_smooth


# ---------------------------------------------------------------------
# 2. System matrices builder
# ---------------------------------------------------------------------

def build_system_matrices(params: np.ndarray,
                          y: np.ndarray,
                          pi: np.ndarray,
                          r_real: np.ndarray):
    """
    Given parameters and data, build F, H, Q, R, c_t, x0, P0, and c.

    params:
      [0]  phi1
      [1]  gamma
      [2]  alpha
      [3]  kappa
      [4]  c
      [5]  log_sigma_y
      [6]  log_sigma_pi
      [7]  log_sigma_y_star
      [8]  log_sigma_g
      [9]  log_sigma_z
      [10] log_meas_y
      [11] log_meas_pi
    """
    phi1, gamma, alpha, kappa, c = params[:5]
    log_sig_y, log_sig_pi, log_sig_ystar, log_sig_g, log_sig_z = params[5:10]
    log_meas_y, log_meas_pi = params[10:12]

    sig_y = np.exp(log_sig_y)
    sig_pi = np.exp(log_sig_pi)
    sig_ystar = np.exp(log_sig_ystar)
    sig_g = np.exp(log_sig_g)
    sig_z = np.exp(log_sig_z)
    meas_y = np.exp(log_meas_y)
    meas_pi = np.exp(log_meas_pi)

    n_states = 7
    n_obs = 2
    T = len(y)

    F = np.zeros((n_states, n_states))

    # State indexing:
    # 0: y_gap_t
    # 1: y_gap_{t-1}
    # 2: pi_t
    # 3: pi_{t-1}
    # 4: y_star_t
    # 5: g_t
    # 6: z_t

    # Output gap transition
    F[0, 0] = phi1            # y_gap_{t-1}
    F[0, 5] = -gamma * c      # g_{t-1}
    F[0, 6] = -gamma          # z_{t-1}
    F[1, 0] = 1.0             # lag: y_gap_{t-1} <- y_gap_t

    # Inflation transition: pi_t = alpha*pi_{t-1} + kappa*y_gap_{t-1} + eps_pi
    F[2, 2] = alpha           # pi_{t-1}
    F[2, 0] = kappa           # y_gap_{t-1}
    F[3, 2] = 1.0             # lag: pi_{t-1} <- pi_t

    # Potential output & trend
    F[4, 4] = 1.0             # y_star_{t-1}
    F[4, 5] = 1.0             # + g_{t-1}
    F[5, 5] = 1.0             # g_t = g_{t-1} + eps_g
    F[6, 6] = 1.0             # z_t = z_{t-1} + eps_z

    # State covariance Q
    Q = np.zeros((n_states, n_states))
    Q[0, 0] = sig_y**2
    Q[2, 2] = sig_pi**2
    Q[4, 4] = sig_ystar**2
    Q[5, 5] = sig_g**2
    Q[6, 6] = sig_z**2

    # Measurement matrix H: [y_obs, pi_obs]
    H = np.zeros((n_obs, n_states))
    # y_obs = y_star + y_gap/100
    H[0, 0] = 1.0 / 100.0
    H[0, 4] = 1.0
    # pi_obs = pi_t
    H[1, 2] = 1.0

    # Measurement covariance R
    R = np.zeros((n_obs, n_obs))
    R[0, 0] = meas_y**2
    R[1, 1] = meas_pi**2

    # Time-varying intercept: gamma * r_{t-1} in y_gap equation
    c_t = np.zeros((T, n_states))
    r_lag = np.empty_like(r_real)
    r_lag[0] = r_real[0]
    r_lag[1:] = r_real[:-1]
    c_t[:, 0] = gamma * r_lag

    # Initial state
    x0 = np.zeros(n_states)
    x0[4] = y[0]     # y_star_0 ~ first y
    x0[2] = pi[0]    # pi_0

    P0 = np.eye(n_states) * 10.0

    return F, H, Q, R, c_t, x0, P0, c


# ---------------------------------------------------------------------
# 3. Bayesian wrapper: PyMC + custom Kalman likelihood
# ---------------------------------------------------------------------

import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

# Data globals for the PyMC op
y_data = None
pi_data = None
r_real_data = None

@as_op(itypes=[pt.dvector], otypes=[pt.dscalar])
def kalman_loglik(params_vec):
    """
    PyTensor-wrapped log-likelihood for PyMC.

    params_vec: flat vector of parameters in the same order as build_system_matrices expects.
    Returns: scalar log-likelihood.
    """
    params = np.array(params_vec, dtype=float)
    F, H, Q, R, c_t, x0, P0, _ = build_system_matrices(params, y_data, pi_data, r_real_data)
    y_stack = np.column_stack([y_data, pi_data])
    kf_res = kalman_filter(y_stack, F, H, Q, R, c_t, x0, P0)
    return np.array(kf_res.loglik, dtype=float)


def run_bayesian_lw(y: np.ndarray,
                    pi: np.ndarray,
                    r_real: np.ndarray,
                    draws: int = 1000,
                    tune: int = 1000):
    """
    Run Bayesian estimation of LW-style model via PyMC.

    Returns:
      trace: PyMC trace object
      posterior_means: dict of posterior mean parameters
      smoothed_state: smoothed state paths for posterior-mean params
      c_hat: posterior-mean link from trend growth to r*
    """
    global y_data, pi_data, r_real_data
    y_data = y
    pi_data = pi
    r_real_data = r_real

    with pm.Model() as model:
        # --- Dynamic coefficients with literature-inspired priors ---

        # Output-gap persistence: high but stationary
        phi1 = pm.TruncatedNormal(
            "phi1", mu=0.8, sigma=0.2, lower=-0.99, upper=0.99
        )

        # Inflation persistence: high but stationary
        alpha = pm.TruncatedNormal(
            "alpha", mu=0.7, sigma=0.2, lower=-0.99, upper=0.99
        )

        # Real-rate gap slope: negative, loose
        gamma = pm.TruncatedNormal(
            "gamma", mu=-0.5, sigma=0.5, lower=-2.0, upper=0.0
        )

        # Phillips-curve slope: small, positive
        kappa = pm.HalfNormal("kappa", sigma=0.2)

        # Link from trend growth to r*: roughly one-for-one, positive
        c = pm.TruncatedNormal(
            "c", mu=1.0, sigma=0.5, lower=0.0, upper=3.0
        )

        # --- Innovation std devs: loose priors in log space ---

        # Center around modest values, very wide (Lewis-style "loose")
        log_sigma_y = pm.Normal("log_sigma_y", mu=np.log(0.5), sigma=1.0)
        log_sigma_pi = pm.Normal("log_sigma_pi", mu=np.log(0.5), sigma=1.0)
        log_sigma_y_star = pm.Normal("log_sigma_y_star", mu=np.log(0.1), sigma=1.0)
        log_sigma_g = pm.Normal("log_sigma_g", mu=np.log(0.05), sigma=1.0)
        log_sigma_z = pm.Normal("log_sigma_z", mu=np.log(0.05), sigma=1.0)

        # --- Measurement error std devs: small but nonzero ---

        log_meas_y = pm.Normal("log_meas_y", mu=np.log(0.05), sigma=1.0)
        log_meas_pi = pm.Normal("log_meas_pi", mu=np.log(0.1), sigma=1.0)

        # Stack into parameter vector for the Kalman log-likelihood op
        params_vec = pt.stack([
            phi1, gamma, alpha, kappa, c,
            log_sigma_y, log_sigma_pi, log_sigma_y_star,
            log_sigma_g, log_sigma_z,
            log_meas_y, log_meas_pi
        ])

        # Custom likelihood
        pm.Potential("likelihood", kalman_loglik(params_vec))

        # Sample from posterior
        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=0.9,
            chains=2,
            cores=2,
        )

    # Posterior mean parameters
    posterior_means = {
        var: trace.posterior[var].mean().item()
        for var in [
            "phi1", "gamma", "alpha", "kappa", "c",
            "log_sigma_y", "log_sigma_pi", "log_sigma_y_star",
            "log_sigma_g", "log_sigma_z", "log_meas_y", "log_meas_pi"
        ]
    }

    params_mean_vec = np.array([
        posterior_means["phi1"],
        posterior_means["gamma"],
        posterior_means["alpha"],
        posterior_means["kappa"],
        posterior_means["c"],
        posterior_means["log_sigma_y"],
        posterior_means["log_sigma_pi"],
        posterior_means["log_sigma_y_star"],
        posterior_means["log_sigma_g"],
        posterior_means["log_sigma_z"],
        posterior_means["log_meas_y"],
        posterior_means["log_meas_pi"],
    ])

    # Kalman smoother for posterior-mean params → smoothed states
    F, H, Q, R, c_t, x0, P0, c_hat = build_system_matrices(
        params_mean_vec, y, pi, r_real
    )
    y_stack = np.column_stack([y, pi])
    kf_res = kalman_filter(y_stack, F, H, Q, R, c_t, x0, P0)
    smoothed_state, smoothed_cov = kalman_smoother(F, kf_res)

    return trace, posterior_means, smoothed_state, c_hat


# ---------------------------------------------------------------------
# 4. Example usage with synthetic data (replace with your FRED-prepped df)
# ---------------------------------------------------------------------

def main():
    # Synthetic data so the script runs; replace with real FRED data.
    T = 200
    dates = pd.period_range("1960Q1", periods=T, freq="Q")
    rng = np.random.default_rng(0)

    trend_growth_true = 0.5 / 4.0  # ~0.5% per year in log
    log_gdp = np.cumsum(trend_growth_true + 0.01 * rng.standard_normal(T))
    pi_series = 2.0 + 0.2 * rng.standard_normal(T)    # around 2% ann.
    r_real = 1.5 + 0.5 * rng.standard_normal(T)       # around 1.5% ann.

    df = pd.DataFrame({
        "log_gdp": log_gdp,
        "pi": pi_series,
        "r_real": r_real,
    }, index=dates)

    # With real data, you’d do:
    # y      = df_q["log_gdp"].values
    # pi     = df_q["pi"].values
    # r_real = df_q["r_real"].values
    y = df["log_gdp"].values
    pi = df["pi"].values
    r_real_arr = df["r_real"].values

    # Run Bayesian estimation
    trace, post_means, smoothed_state, c_hat = run_bayesian_lw(
        y, pi, r_real_arr, draws=1000, tune=1000
    )

    # Extract smoothed r* from posterior-mean params
    y_gap_hat = smoothed_state[:, 0]
    y_star_hat = smoothed_state[:, 4]
    g_hat = smoothed_state[:, 5]
    z_hat = smoothed_state[:, 6]
    r_star_hat = c_hat * g_hat + z_hat

    results = pd.DataFrame({
        "y": y,
        "y_star_hat": y_star_hat,
        "y_gap_hat": y_gap_hat,
        "g_hat": g_hat,
        "z_hat": z_hat,
        "r_star_hat": r_star_hat,
        "r_real": r_real_arr,
        "pi": pi,
    }, index=dates)

    print("Posterior mean parameters:")
    for k, v in post_means.items():
        print(f"{k:>14s} = {v: .4f}")

    print("\nFirst few smoothed states (posterior-mean params):")
    print(results.head())

    # Optional plotting
    try:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(results.index.to_timestamp(), results["y"], label="log y")
        axs[0].plot(results.index.to_timestamp(), results["y_star_hat"], label="log y* (posterior-mean)")
        axs[0].legend()
        axs[0].set_title("Output and Potential")

        axs[1].plot(results.index.to_timestamp(), results["y_gap_hat"], label="Output gap")
        axs[1].axhline(0.0, color="k", linewidth=0.5)
        axs[1].legend()
        axs[1].set_title("Output Gap")

        axs[2].plot(results.index.to_timestamp(), results["r_real"], label="Real rate (data)")
        axs[2].plot(results.index.to_timestamp(), results["r_star_hat"], label="r* (posterior-mean params)")
        axs[2].legend()
        axs[2].set_title("Real Rate vs r*")

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not installed; skipping plots.")


if __name__ == "__main__":
    main()
```

You can now just swap the synthetic `df` in `main()` for your FRED-prepped `df_q` (with `log_gdp`, `pi`, `r_real`) and this will give you a Bayesian LW-style r* estimate with literature-consistent priors.
Here’s the **fully updated Bayesian LW-style r*** script with priors adjusted to be more in line with the literature (Lewis-style loose variance priors, SW-style persistence, etc.). No hours worked anywhere.

```python
#!/usr/bin/env python3
"""
Bayesian Laubach-Williams-style r* estimator (no hours).

States (7):
  0: y_gap_t      = 100 * (y_t - y*_t)      # output gap in %
  1: y_gap_{t-1}
  2: pi_t         # inflation
  3: pi_{t-1}
  4: y_star_t     # log potential output
  5: g_t          # trend growth (quarterly, ann.)
  6: z_t          # other component of r*

Transition:
  y_gap_t = phi1 * y_gap_{t-1}
            + gamma * (r_{t-1} - (c * g_{t-1} + z_{t-1})) + eps_y

  y_gap_{t-1} <- y_gap_t (lag)
  pi_t        = alpha * pi_{t-1} + kappa * y_gap_{t-1} + eps_pi
  pi_{t-1}    <- pi_t (lag)

  y_star_t    = y_star_{t-1} + g_{t-1} + eps_y_star
  g_t         = g_{t-1} + eps_g
  z_t         = z_{t-1} + eps_z

Measurement:
  y_obs_t  = y_star_t + y_gap_t / 100 + meas_y_noise
  pi_obs_t = pi_t + meas_pi_noise

We integrate out the states analytically via a Kalman filter
and do Bayesian inference on parameters via PyMC, using
literature-inspired priors to avoid the pile-up-at-zero problem.
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Tuple

# ---------------------------------------------------------------------
# 1. Kalman filter and smoother
# ---------------------------------------------------------------------

@dataclass
class KFResults:
    loglik: float
    filtered_state: np.ndarray
    filtered_cov: np.ndarray
    pred_state: np.ndarray
    pred_cov: np.ndarray


def kalman_filter(
    y: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    c_t: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
) -> KFResults:
    """
    Linear Gaussian Kalman filter with time-varying state intercept c_t.
    """
    T, n_obs = y.shape
    n_states = F.shape[0]

    filtered_state = np.zeros((T, n_states))
    filtered_cov = np.zeros((T, n_states, n_states))
    pred_state = np.zeros((T, n_states))
    pred_cov = np.zeros((T, n_states, n_states))

    x_filt = x0.copy()
    P_filt = P0.copy()
    loglik = 0.0
    two_pi = 2.0 * np.pi

    for t in range(T):
        # Predict
        x_pred = F @ x_filt + c_t[t]
        P_pred = F @ P_filt @ F.T + Q

        # Forecast y
        y_pred = H @ x_pred
        v = y[t] - y_pred
        S = H @ P_pred @ H.T + R

        try:
            sign, logdet = np.linalg.slogdet(S)
        except np.linalg.LinAlgError:
            return KFResults(-1e10, filtered_state, filtered_cov, pred_state, pred_cov)

        if sign <= 0:
            return KFResults(-1e10, filtered_state, filtered_cov, pred_state, pred_cov)

        S_inv = np.linalg.inv(S)
        ll_t = -0.5 * (n_obs * np.log(two_pi) + logdet + v.T @ S_inv @ v)
        loglik += ll_t

        # Update
        K = P_pred @ H.T @ S_inv
        x_filt = x_pred + K @ v
        P_filt = (np.eye(n_states) - K @ H) @ P_pred

        filtered_state[t] = x_filt
        filtered_cov[t] = P_filt
        pred_state[t] = x_pred
        pred_cov[t] = P_pred

    return KFResults(
        loglik=loglik,
        filtered_state=filtered_state,
        filtered_cov=filtered_cov,
        pred_state=pred_state,
        pred_cov=pred_cov,
    )


def kalman_smoother(F: np.ndarray, kf_res: KFResults) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel smoother.
    """
    filtered_state = kf_res.filtered_state
    filtered_cov = kf_res.filtered_cov
    pred_state = kf_res.pred_state
    pred_cov = kf_res.pred_cov

    T, n_states = filtered_state.shape
    x_smooth = np.zeros_like(filtered_state)
    P_smooth = np.zeros_like(filtered_cov)

    # Start from last
    x_smooth[-1] = filtered_state[-1]
    P_smooth[-1] = filtered_cov[-1]

    for t in range(T - 2, -1, -1):
        P_filt = filtered_cov[t]
        P_pred_next = pred_cov[t + 1]
        C = P_filt @ F.T @ np.linalg.inv(P_pred_next)
        x_smooth[t] = filtered_state[t] + C @ (x_smooth[t + 1] - pred_state[t + 1])
        P_smooth[t] = filtered_cov[t] + C @ (P_smooth[t + 1] - P_pred_next) @ C.T

    return x_smooth, P_smooth


# ---------------------------------------------------------------------
# 2. System matrices builder
# ---------------------------------------------------------------------

def build_system_matrices(params: np.ndarray,
                          y: np.ndarray,
                          pi: np.ndarray,
                          r_real: np.ndarray):
    """
    Given parameters and data, build F, H, Q, R, c_t, x0, P0, and c.

    params:
      [0]  phi1
      [1]  gamma
      [2]  alpha
      [3]  kappa
      [4]  c
      [5]  log_sigma_y
      [6]  log_sigma_pi
      [7]  log_sigma_y_star
      [8]  log_sigma_g
      [9]  log_sigma_z
      [10] log_meas_y
      [11] log_meas_pi
    """
    phi1, gamma, alpha, kappa, c = params[:5]
    log_sig_y, log_sig_pi, log_sig_ystar, log_sig_g, log_sig_z = params[5:10]
    log_meas_y, log_meas_pi = params[10:12]

    sig_y = np.exp(log_sig_y)
    sig_pi = np.exp(log_sig_pi)
    sig_ystar = np.exp(log_sig_ystar)
    sig_g = np.exp(log_sig_g)
    sig_z = np.exp(log_sig_z)
    meas_y = np.exp(log_meas_y)
    meas_pi = np.exp(log_meas_pi)

    n_states = 7
    n_obs = 2
    T = len(y)

    F = np.zeros((n_states, n_states))

    # State indexing:
    # 0: y_gap_t
    # 1: y_gap_{t-1}
    # 2: pi_t
    # 3: pi_{t-1}
    # 4: y_star_t
    # 5: g_t
    # 6: z_t

    # Output gap transition
    F[0, 0] = phi1            # y_gap_{t-1}
    F[0, 5] = -gamma * c      # g_{t-1}
    F[0, 6] = -gamma          # z_{t-1}
    F[1, 0] = 1.0             # lag: y_gap_{t-1} <- y_gap_t

    # Inflation transition: pi_t = alpha*pi_{t-1} + kappa*y_gap_{t-1} + eps_pi
    F[2, 2] = alpha           # pi_{t-1}
    F[2, 0] = kappa           # y_gap_{t-1}
    F[3, 2] = 1.0             # lag: pi_{t-1} <- pi_t

    # Potential output & trend
    F[4, 4] = 1.0             # y_star_{t-1}
    F[4, 5] = 1.0             # + g_{t-1}
    F[5, 5] = 1.0             # g_t = g_{t-1} + eps_g
    F[6, 6] = 1.0             # z_t = z_{t-1} + eps_z

    # State covariance Q
    Q = np.zeros((n_states, n_states))
    Q[0, 0] = sig_y**2
    Q[2, 2] = sig_pi**2
    Q[4, 4] = sig_ystar**2
    Q[5, 5] = sig_g**2
    Q[6, 6] = sig_z**2

    # Measurement matrix H: [y_obs, pi_obs]
    H = np.zeros((n_obs, n_states))
    # y_obs = y_star + y_gap/100
    H[0, 0] = 1.0 / 100.0
    H[0, 4] = 1.0
    # pi_obs = pi_t
    H[1, 2] = 1.0

    # Measurement covariance R
    R = np.zeros((n_obs, n_obs))
    R[0, 0] = meas_y**2
    R[1, 1] = meas_pi**2

    # Time-varying intercept: gamma * r_{t-1} in y_gap equation
    c_t = np.zeros((T, n_states))
    r_lag = np.empty_like(r_real)
    r_lag[0] = r_real[0]
    r_lag[1:] = r_real[:-1]
    c_t[:, 0] = gamma * r_lag

    # Initial state
    x0 = np.zeros(n_states)
    x0[4] = y[0]     # y_star_0 ~ first y
    x0[2] = pi[0]    # pi_0

    P0 = np.eye(n_states) * 10.0

    return F, H, Q, R, c_t, x0, P0, c


# ---------------------------------------------------------------------
# 3. Bayesian wrapper: PyMC + custom Kalman likelihood
# ---------------------------------------------------------------------

import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

# Data globals for the PyMC op
y_data = None
pi_data = None
r_real_data = None

@as_op(itypes=[pt.dvector], otypes=[pt.dscalar])
def kalman_loglik(params_vec):
    """
    PyTensor-wrapped log-likelihood for PyMC.

    params_vec: flat vector of parameters in the same order as build_system_matrices expects.
    Returns: scalar log-likelihood.
    """
    params = np.array(params_vec, dtype=float)
    F, H, Q, R, c_t, x0, P0, _ = build_system_matrices(params, y_data, pi_data, r_real_data)
    y_stack = np.column_stack([y_data, pi_data])
    kf_res = kalman_filter(y_stack, F, H, Q, R, c_t, x0, P0)
    return np.array(kf_res.loglik, dtype=float)


def run_bayesian_lw(y: np.ndarray,
                    pi: np.ndarray,
                    r_real: np.ndarray,
                    draws: int = 1000,
                    tune: int = 1000):
    """
    Run Bayesian estimation of LW-style model via PyMC.

    Returns:
      trace: PyMC trace object
      posterior_means: dict of posterior mean parameters
      smoothed_state: smoothed state paths for posterior-mean params
      c_hat: posterior-mean link from trend growth to r*
    """
    global y_data, pi_data, r_real_data
    y_data = y
    pi_data = pi
    r_real_data = r_real

    with pm.Model() as model:
        # --- Dynamic coefficients with literature-inspired priors ---

        # Output-gap persistence: high but stationary
        phi1 = pm.TruncatedNormal(
            "phi1", mu=0.8, sigma=0.2, lower=-0.99, upper=0.99
        )

        # Inflation persistence: high but stationary
        alpha = pm.TruncatedNormal(
            "alpha", mu=0.7, sigma=0.2, lower=-0.99, upper=0.99
        )

        # Real-rate gap slope: negative, loose
        gamma = pm.TruncatedNormal(
            "gamma", mu=-0.5, sigma=0.5, lower=-2.0, upper=0.0
        )

        # Phillips-curve slope: small, positive
        kappa = pm.HalfNormal("kappa", sigma=0.2)

        # Link from trend growth to r*: roughly one-for-one, positive
        c = pm.TruncatedNormal(
            "c", mu=1.0, sigma=0.5, lower=0.0, upper=3.0
        )

        # --- Innovation std devs: loose priors in log space ---

        # Center around modest values, very wide (Lewis-style "loose")
        log_sigma_y = pm.Normal("log_sigma_y", mu=np.log(0.5), sigma=1.0)
        log_sigma_pi = pm.Normal("log_sigma_pi", mu=np.log(0.5), sigma=1.0)
        log_sigma_y_star = pm.Normal("log_sigma_y_star", mu=np.log(0.1), sigma=1.0)
        log_sigma_g = pm.Normal("log_sigma_g", mu=np.log(0.05), sigma=1.0)
        log_sigma_z = pm.Normal("log_sigma_z", mu=np.log(0.05), sigma=1.0)

        # --- Measurement error std devs: small but nonzero ---

        log_meas_y = pm.Normal("log_meas_y", mu=np.log(0.05), sigma=1.0)
        log_meas_pi = pm.Normal("log_meas_pi", mu=np.log(0.1), sigma=1.0)

        # Stack into parameter vector for the Kalman log-likelihood op
        params_vec = pt.stack([
            phi1, gamma, alpha, kappa, c,
            log_sigma_y, log_sigma_pi, log_sigma_y_star,
            log_sigma_g, log_sigma_z,
            log_meas_y, log_meas_pi
        ])

        # Custom likelihood
        pm.Potential("likelihood", kalman_loglik(params_vec))

        # Sample from posterior
        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=0.9,
            chains=2,
            cores=2,
        )

    # Posterior mean parameters
    posterior_means = {
        var: trace.posterior[var].mean().item()
        for var in [
            "phi1", "gamma", "alpha", "kappa", "c",
            "log_sigma_y", "log_sigma_pi", "log_sigma_y_star",
            "log_sigma_g", "log_sigma_z", "log_meas_y", "log_meas_pi"
        ]
    }

    params_mean_vec = np.array([
        posterior_means["phi1"],
        posterior_means["gamma"],
        posterior_means["alpha"],
        posterior_means["kappa"],
        posterior_means["c"],
        posterior_means["log_sigma_y"],
        posterior_means["log_sigma_pi"],
        posterior_means["log_sigma_y_star"],
        posterior_means["log_sigma_g"],
        posterior_means["log_sigma_z"],
        posterior_means["log_meas_y"],
        posterior_means["log_meas_pi"],
    ])

    # Kalman smoother for posterior-mean params → smoothed states
    F, H, Q, R, c_t, x0, P0, c_hat = build_system_matrices(
        params_mean_vec, y, pi, r_real
    )
    y_stack = np.column_stack([y, pi])
    kf_res = kalman_filter(y_stack, F, H, Q, R, c_t, x0, P0)
    smoothed_state, smoothed_cov = kalman_smoother(F, kf_res)

    return trace, posterior_means, smoothed_state, c_hat


# ---------------------------------------------------------------------
# 4. Example usage with synthetic data (replace with your FRED-prepped df)
# ---------------------------------------------------------------------

def main():
    # Synthetic data so the script runs; replace with real FRED data.
    T = 200
    dates = pd.period_range("1960Q1", periods=T, freq="Q")
    rng = np.random.default_rng(0)

    trend_growth_true = 0.5 / 4.0  # ~0.5% per year in log
    log_gdp = np.cumsum(trend_growth_true + 0.01 * rng.standard_normal(T))
    pi_series = 2.0 + 0.2 * rng.standard_normal(T)    # around 2% ann.
    r_real = 1.5 + 0.5 * rng.standard_normal(T)       # around 1.5% ann.

    df = pd.DataFrame({
        "log_gdp": log_gdp,
        "pi": pi_series,
        "r_real": r_real,
    }, index=dates)

    # With real data, you’d do:
    # y      = df_q["log_gdp"].values
    # pi     = df_q["pi"].values
    # r_real = df_q["r_real"].values
    y = df["log_gdp"].values
    pi = df["pi"].values
    r_real_arr = df["r_real"].values

    # Run Bayesian estimation
    trace, post_means, smoothed_state, c_hat = run_bayesian_lw(
        y, pi, r_real_arr, draws=1000, tune=1000
    )

    # Extract smoothed r* from posterior-mean params
    y_gap_hat = smoothed_state[:, 0]
    y_star_hat = smoothed_state[:, 4]
    g_hat = smoothed_state[:, 5]
    z_hat = smoothed_state[:, 6]
    r_star_hat = c_hat * g_hat + z_hat

    results = pd.DataFrame({
        "y": y,
        "y_star_hat": y_star_hat,
        "y_gap_hat": y_gap_hat,
        "g_hat": g_hat,
        "z_hat": z_hat,
        "r_star_hat": r_star_hat,
        "r_real": r_real_arr,
        "pi": pi,
    }, index=dates)

    print("Posterior mean parameters:")
    for k, v in post_means.items():
        print(f"{k:>14s} = {v: .4f}")

    print("\nFirst few smoothed states (posterior-mean params):")
    print(results.head())

    # Optional plotting
    try:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(results.index.to_timestamp(), results["y"], label="log y")
        axs[0].plot(results.index.to_timestamp(), results["y_star_hat"], label="log y* (posterior-mean)")
        axs[0].legend()
        axs[0].set_title("Output and Potential")

        axs[1].plot(results.index.to_timestamp(), results["y_gap_hat"], label="Output gap")
        axs[1].axhline(0.0, color="k", linewidth=0.5)
        axs[1].legend()
        axs[1].set_title("Output Gap")

        axs[2].plot(results.index.to_timestamp(), results["r_real"], label="Real rate (data)")
        axs[2].plot(results.index.to_timestamp(), results["r_star_hat"], label="r* (posterior-mean params)")
        axs[2].legend()
        axs[2].set_title("Real Rate vs r*")

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib not installed; skipping plots.")


if __name__ == "__main__":
    main()
