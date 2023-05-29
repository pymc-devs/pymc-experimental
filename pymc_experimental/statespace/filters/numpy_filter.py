from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg

MVN_CONST = np.log(2 * np.pi)


def build_mask_matrix(nan_mask: ArrayLike) -> ArrayLike:
    """
    The Kalman Filter can "natively" handle missing values by treating observed states as un-observed states for
    iterations where data is not available. To do this, the Z and H matrices must be modified. This function creates
    a matrix W such that W @ Z and W @ H have zeros where data is missing.
    Parameters
    ----------
    nan_mask: array
        A 1d array of boolean flags of length n, indicating whether a state is observed in the current iteration.
    Returns
    -------
    W: array
        An n x n matrix used to mask missing values in the Z and H matrices
    """
    n = nan_mask.shape[0]
    W = np.eye(n)
    i = 0
    for flag in nan_mask:
        if flag:
            W[i, i] = 0
        i += 1

    W = np.ascontiguousarray(W)

    return W


def standard_kalman_filter(
    data: ArrayLike,
    c: ArrayLike,
    d: ArrayLike,
    T: ArrayLike,
    Z: ArrayLike,
    R: ArrayLike,
    H: ArrayLike,
    Q: ArrayLike,
    a0: ArrayLike,
    P0: ArrayLike,
) -> Tuple:
    """
    Parameters
    ----------
    data: array
        (T, k_observed) matrix of observed data. Data can include missing values.
    c: array
        (k_states, 1, *) matrix of bias terms on transition equations. Last dimension should be 1 if biases are not
        time varying, else T.
    d: array
        (k_observed, 1, *) matrix of bias terms on observations equations. Last dimension should be 1 if biases are not
        time varying, else T.
    a0: array
        (k_states, 1) vector of initial states.
    P0: array
        (k_states, k_states) initial state covariance matrix
    T: array
        (k_states, k_states, *) transition matrix
    Z: array
        (k_states, k_observed, *) design matrix
    R: array
    H: array
    Q: array
    Returns
    -------
    """
    n_steps, k_obs, *_ = data.shape
    k_states, k_posdef = R.shape

    filtered_states = np.zeros((n_steps, k_states, 1))
    predicted_states = np.zeros((n_steps + 1, k_states, 1))
    filtered_cov = np.zeros((n_steps, k_states, k_states))
    predicted_cov = np.zeros((n_steps + 1, k_states, k_states))
    log_likelihood = np.zeros(n_steps)

    a = a0
    P = P0

    predicted_states[0] = a
    predicted_cov[0] = P

    for i in range(n_steps):
        a_filtered, a_hat, P_filtered, P_hat, ll = kalman_step(data[i], a, P, c, d, T, Z, R, H, Q)

        filtered_states[i] = a_filtered
        predicted_states[i + 1] = a_hat
        filtered_cov[i] = P_filtered
        predicted_cov[i + 1] = P_hat
        log_likelihood[i] = ll[0]

        a = a_hat
        P = P_hat

    return (
        filtered_states,
        predicted_states,
        filtered_cov,
        predicted_cov,
        log_likelihood,
    )


def kalman_step(y, a, P, c, d, T, Z, R, H, Q):
    nan_mask = np.isnan(y)
    all_nan_flag = np.all(nan_mask)

    W = build_mask_matrix(nan_mask)

    Z_masked = W @ Z
    H_masked = W @ H
    y_masked = y.copy()
    y_masked[nan_mask] = 0.0

    a_filtered, P_filtered, ll = filter_step(a, P, y_masked, d, Z_masked, H_masked, all_nan_flag)

    a_hat, P_hat = predict(a=a_filtered, P=P_filtered, c=c, T=T, R=R, Q=Q)

    return a_filtered, a_hat, P_filtered, P_hat, ll


def filter_step(a, P, y, d, Z, H, all_nan_flag):
    if all_nan_flag:
        # If everything is missing, no filtering is necessary

        a_filtered = np.atleast_2d(a).reshape((-1, 1))
        P_filtered = P
        ll = np.zeros(y.shape[0])

        return a_filtered, P_filtered, ll

    v = y - Z @ a - d

    PZT = P @ Z.T
    F = Z @ PZT + H

    # TODO: Benchmark this double triangular method against linalg.solve
    F_chol = np.linalg.cholesky(F)
    K = linalg.solve_triangular(
        F_chol, linalg.solve_triangular(F_chol, PZT.T, lower=True), trans=1, lower=True
    ).T

    I_KZ = np.eye(K.shape[0]) - K @ Z

    a_filtered = a + K @ v
    P_filtered = I_KZ @ P @ I_KZ.T + K @ H @ K.T
    P_filtered = 0.5 * (P_filtered + P_filtered.T)

    inner_term = linalg.solve_triangular(
        F_chol, linalg.solve_triangular(F_chol, v, lower=True), lower=True, trans=1
    )
    n = y.shape[0]
    ll = -0.5 * (n * MVN_CONST + (v.T @ inner_term).ravel()) - np.log(np.diag(F_chol)).sum()

    return a_filtered, P_filtered, ll


def predict(a, P, c, T, R, Q):
    a_hat = T @ a + c
    P_hat = T @ P @ T.T + R @ Q @ R.T

    return a_hat, P_hat


def kalman_filter(data, a0, P0, c, d, T, Z, R, H, Q):
    filter_results = standard_kalman_filter(data, c, d, T, Z, R, H, Q, a0, P0)

    return filter_results
