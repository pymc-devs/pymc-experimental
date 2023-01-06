#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import aesara.tensor as at
from aesara.tensor.var import TensorVariable
import aeppl


def kalman_filter_onestep(
    y_t_val: TensorVariable,
    Ft: TensorVariable,
    Qt: TensorVariable,
    Ht: TensorVariable,
    Rt: TensorVariable,
    mu_t_tm1: TensorVariable,
    P_t_tm1: TensorVariable,
):
    """Perform a single step of the Kalman filter.

    A naive implementation of the Kalman filter, which migth not be numerically stable.
    Notation follows [1].

    Args:
        y_t_val: The observed value at time t.
        Ft: The transition matrix at time t.
        Qt: The transition covariance at time t.
        Ht: The observation matrix at time t.
        Rt: The observation covariance at time t.
        mu_t_tm1: The predicted mean of the state at time t-1.
        P_t_tm1: The predicted covariance of the state at time t-1.
    Returns:
        mu_tp1_t: The predicted mean of the state at time t+1.
        P_tp1_t: The predicted covariance of the state at time t+1.
        mu_t_t: The filtered mean of the state at time t.
        P_t_t: The filtered covariance of the state at time t.
        y_hat_t: The predicted observation mean at time t.
        St: The predicted observation covariance at time t.
        Y_t_logp: The logp of the observation at time t.

    References:
    ----------
    .. [1] Martin, Osvaldo A., Ravin Kumar, and Junpeng Lao. 
        Bayesian Modeling and Computation in Python (Chapter 6). 
        Chapman and Hall/CRC, 2022.
    """
    # Observation
    y_hat_t = Ht @ mu_t_tm1
    St = Ht @ P_t_tm1 @ Ht.T + Rt
    Y_t = at.random.multivariate_normal(y_hat_t, St, name="Y_t")
    Y_t_logp = aeppl.logprob(Y_t, y_t_val)
    Y_t_logp.name = "log(Y_t=y_t)"

    # Optimal Kalman gain K_t
    Kt = P_t_tm1 @ Ht.T @ (St**-1)

    # Compute the filtered state
    mu_t_t = mu_t_tm1 + Kt @ (y_t_val - y_hat_t)
    # P* = P - K * H * P
    P_t_t = P_t_tm1 - Kt @ Ht @ P_t_tm1
    # P_t_t = P_t_tm1 - K_t @ S_t @ K_t.T

    # Compute the predicted state
    mu_tp1_t = Ft @ mu_t_t
    P_tp1_t = Ft @ P_t_t @ Ft.T + Qt
    return mu_tp1_t, P_tp1_t, mu_t_t, P_t_t, y_hat_t, St, Y_t_logp
