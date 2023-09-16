from abc import ABC
from typing import List, Optional, Tuple

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.compile.mode import get_mode
from pytensor.graph.basic import Variable
from pytensor.raise_op import Assert
from pytensor.tensor import TensorVariable
from pytensor.tensor.nlinalg import matrix_dot
from pytensor.tensor.slinalg import solve_triangular

from pymc_experimental.statespace.filters.utilities import (
    quad_form_sym,
    split_vars_into_seq_and_nonseq,
    stabilize,
)
from pymc_experimental.statespace.utils.constants import JITTER_DEFAULT, MISSING_FILL
from pymc_experimental.statespace.utils.pytensor_scipy import solve_discrete_are

MVN_CONST = pt.log(2 * pt.constant(np.pi, dtype="float64"))
PARAM_NAMES = ["c", "d", "T", "Z", "R", "H", "Q"]

assert_data_is_1d = Assert("UnivariateTimeSeries filter requires data be at most 1-dimensional")
assert_time_varying_dim_correct = Assert(
    "The first dimension of a time varying matrix (the time dimension) must be "
    "equal to the first dimension of the data (the time dimension)."
)


class BaseFilter(ABC):
    def __init__(self, mode=None):
        """
        Kalman Filter.

        Parameters
        ----------
        mode : str, optional
            The mode used for Pytensor compilation. Defaults to None.

        Notes
        -----
        The BaseFilter class is an abstract base class (ABC) for implementing kalman filters.
        It defines common attributes and methods used by kalman filter implementations.

        Attributes
        ----------
        mode : str or None
            The mode used for Pytensor compilation.

        seq_names : List[str]
            A list of name representing time-varying statespace matrices. That is, inputs that will need to be
            provided to the `sequences` argument of `pytensor.scan`

        non_seq_names : List[str]
            A list of names representing static statespace matrices. That is, inputs that will need to be provided
            to the `non_sequences` argument of `pytensor.scan`

        eye_states : TensorVariable
            An identity matrix of shape (k_states, k_states), stored for computational efficiency

        eye_posdef : TensorVariable
            An identity matrix of shape (k_posdef, k_posdef), stored for computational efficiency

        eye_endog : TensorVariable
            An identity matrix of shape (k_endog, k_endog), stored for computational efficiency
        """

        self.mode: str = mode
        self.seq_names: List[str] = []
        self.non_seq_names: List[str] = []

        self.n_states = None
        self.n_posdef = None
        self.n_endog = None

        self.eye_states: Optional[TensorVariable] = None
        self.eye_posdef: Optional[TensorVariable] = None
        self.eye_endog: Optional[TensorVariable] = None
        self.missing_fill_value: Optional[float] = None
        self.cov_jitter = None

    def initialize_eyes(self, R: TensorVariable, Z: TensorVariable) -> None:
        """
        Initialize identity matrices for of shapes repeated used in the kalman filtering equations and store them.

        It's surprisingly expensive for pytensor to create an identity matrix every time we need one
        (see [1] for benchmarks). This function creates some identity matrices of useful sizes for the model
        to re-use as a small optimization.

        Parameters
        ----------
        R : TensorVariable
            The tensor representing the selection matrix, called R in [2]

        Z : TensorVariable
            The tensor representing the design matrix, called Z in [2].

        Returns
        -------
        None

        References
        ----------
        .. [1] https://gist.github.com/jessegrabowski/acd3235833163943a11654d78a72f04b
        .. [2] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        """

        self.n_states, self.n_posdef, self.n_endog = R.shape[-2], R.shape[-1], Z.shape[-2]
        self.eye_states = pt.eye(self.n_states)
        self.eye_posdef = pt.eye(self.n_posdef)
        self.eye_endog = pt.eye(self.n_endog)

    def check_params(self, data, a0, P0, c, d, T, Z, R, H, Q):
        """
        Apply any checks on validity of inputs. For most filters this is just the identity function.
        """
        return data, a0, P0, c, d, T, Z, R, H, Q

    @staticmethod
    def add_check_on_time_varying_shapes(
        data: TensorVariable, sequence_params: List[TensorVariable]
    ) -> List[Variable]:
        """
        Insert a check that time-varying matrices match the data shape to the computational graph.

        If any matrices are time-varying, they need to have the same length as the data. This function wraps each
        element of `sequence_params` in an assert `Op` that makes sure all inputs have the correct shape.

        Parameters
        ----------
        data : TensorVariable
            The tensor representing the data.

        sequence_params : List[TensorVariable]
            A list of tensors to be provided to `pytensor.scan` as `sequences`.

        Returns
        -------
        List[TensorVariable]
            A list of tensors wrapped in an `Assert` `Op` that checks the shape of the 0th dimension on each is equal
             to the shape of the 0th dimension on the data.

        # TODO: The PytensorRepresentation object puts the time dimension last, should the reshaping happen here in
            the Kalman filter, or in the StateSpaceModel, before passing into the KF?
        """
        params_with_assert = [
            assert_time_varying_dim_correct(param, pt.eq(param.shape[0], data.shape[0]))
            for param in sequence_params
        ]

        return params_with_assert

    def unpack_args(self, args) -> Tuple:
        """
        The order of inputs to the inner scan function is not known, since some, all, or none of the input matrices
        can be time varying. The order arguments are fed to the inner function is sequences, outputs_info,
        non-sequences. This function works out which matrices are where, and returns a standardized order expected
        by the kalman_step function.

        The standard order is: y, a0, P0, c, d, T, Z, R, H, Q
        """
        # If there are no sequence parameters (all params are static),
        # no changes are needed, params will be in order.
        args = list(args)
        n_seq = len(self.seq_names)
        if n_seq == 0:
            return args

        # The first arg is always y
        y = args.pop(0)

        # There are always two outputs_info wedged between the seqs and non_seqs
        seqs, (a0, P0), non_seqs = args[:n_seq], args[n_seq : n_seq + 2], args[n_seq + 2 :]
        return_ordered = []
        for name in ["c", "d", "T", "Z", "R", "H", "Q"]:
            if name in self.seq_names:
                idx = self.seq_names.index(name)
                return_ordered.append(seqs[idx])
            else:
                idx = self.non_seq_names.index(name)
                return_ordered.append(non_seqs[idx])

        c, d, T, Z, R, H, Q = return_ordered

        return y, a0, P0, c, d, T, Z, R, H, Q

    def build_graph(
        self,
        data,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        mode=None,
        return_updates=False,
        missing_fill_value=None,
        cov_jitter=None,
    ) -> List[TensorVariable]:
        """
        Construct the computation graph for the Kalman filter. See [1] for details.

        Parameters
        ----------
        data : TensorVariable
            Data to be filtered

        mode : optional, str
            Pytensor compile mode, passed to pytensor.scan

        return_updates: bool, default False
            Whether to return updates associated with the pytensor scan. Should only be requried to debug pruposes.

        missing_fill_value: float, default -9999
            Fill value used to mark missing values. Used to avoid PyMC's automatic interpolation, which conflict's with
            the Kalman filter's hidden state inference. Change if your data happens to have legitimate values of -9999

        cov_jitter: float, default 1e-8 or 1e-6 if pytensor.config.floatX is float32
            The Kalman filter is known to be numerically unstable, especially at half precision. This value is added to
            the diagonal of every covariance matrix -- predicted, filtered, and smoothed -- at every step, to ensure
            all matrices are strictly positive semi-definite.

            Obviously, if this can be zero, that's best. In general:
                - Having measurement error makes Kalman Filters more robust. A large source of numerical errors come
                  from the Filtered and Smoothed matrices having a zero in the (0, 0) position, which always occurs
                  when there is no measurement error.

                - The Univariate Filter is more robust than other filters, and can tolerate a lower jitter value

        References
        ----------
        .. [1] Koopman, Siem Jan, Neil Shephard, and Jurgen A. Doornik. 1999.
           Statistical Algorithms for Models in State Space Using SsfPack 2.2.
           Econometrics Journal 2 (1): 107-60. doi:10.1111/1368-423X.00023.
        """
        if missing_fill_value is None:
            missing_fill_value = MISSING_FILL
        if cov_jitter is None:
            cov_jitter = JITTER_DEFAULT

        self.mode = mode
        self.missing_fill_value = missing_fill_value
        self.initialize_eyes(R, Z)
        self.cov_jitter = cov_jitter

        data, a0, P0, *params = self.check_params(data, a0, P0, c, d, T, Z, R, H, Q)

        sequences, non_sequences, seq_names, non_seq_names = split_vars_into_seq_and_nonseq(
            params, PARAM_NAMES
        )

        self.seq_names = seq_names
        self.non_seq_names = non_seq_names

        if len(sequences) > 0:
            sequences = self.add_check_on_time_varying_shapes(data, sequences)

        results, updates = pytensor.scan(
            self.kalman_step,
            sequences=[data] + sequences,
            outputs_info=[None, a0, None, None, P0, None, None],
            non_sequences=non_sequences,
            name="forward_kalman_pass",
            mode=get_mode(self.mode),
            strict=False,
        )

        filter_results = self._postprocess_scan_results(results, a0, P0, n=data.type.shape[0])

        if return_updates:
            return filter_results, updates
        return filter_results

    def _postprocess_scan_results(self, results, a0, P0, n) -> List[TensorVariable]:
        """
        Transform the values returned by the Kalman Filter scan into a form expected by users. In particular:
        1. Append the initial state and covariance matrix to their respective Kalman predictions. This matches the
        output returned by Statsmodels state space models.

        2. Discard the last state and covariance matrix from the Kalman predictions. This is beacuse the kalman filter
        starts with the (random variable) initial state x0, and treats it as a predicted state. The first step (t=0)
        will filter x0 to make filtered_states[0], then do a predict step to make predicted_states[1]. This means
        the last step (t=T) predicted state will be a *forecast* for T+1. If the user wants this forecast, he should
        use the forecast method.

        3. Squeeze away extra dimensions from the filtered and predicted states, as well as the likelihoods.
        """
        (
            filtered_states,
            predicted_states,
            observed_states,
            filtered_covariances,
            predicted_covariances,
            observed_covariances,
            loglike_obs,
        ) = results

        predicted_states = pt.concatenate(
            [pt.expand_dims(a0, axis=(0,)), predicted_states[:-1]], axis=0
        )
        predicted_covariances = pt.concatenate(
            [pt.expand_dims(P0, axis=(0,)), predicted_covariances[:-1]], axis=0
        )

        filtered_states = pt.specify_shape(filtered_states, (n, self.n_states))
        filtered_states.name = "filtered_states"

        predicted_states = pt.specify_shape(predicted_states, (n, self.n_states))
        predicted_states.name = "predicted_states"

        observed_states = pt.specify_shape(observed_states, (n, self.n_endog))
        observed_states.name = "observed_states"

        filtered_covariances = pt.specify_shape(
            filtered_covariances, (n, self.n_states, self.n_states)
        )
        filtered_covariances.name = "filtered_covariances"

        predicted_covariances = pt.specify_shape(
            predicted_covariances, (n, self.n_states, self.n_states)
        )
        predicted_covariances.name = "predicted_covariances"

        observed_covariances = pt.specify_shape(
            observed_covariances, (n, self.n_endog, self.n_endog)
        )
        observed_covariances.name = "observed_covariances"

        loglike_obs = pt.specify_shape(loglike_obs.squeeze(), (n,))
        loglike_obs.name = "loglike_obs"

        filter_results = [
            filtered_states,
            predicted_states,
            observed_states,
            filtered_covariances,
            predicted_covariances,
            observed_covariances,
            loglike_obs,
        ]

        return filter_results

    def handle_missing_values(
        self, y, Z, H
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable, float]:
        """
        This function handles missing values in the observation data `y` and adjusts the design matrix `Z` and the
        observation noise covariance matrix `H` accordingly. Missing values are replaced with zeros to prevent
        propagating NaNs through the computation. The function also returns a binary flag tensor `all_nan_flag`,
        indicating if all values in the observation data are missing. This flag is used for numerical adjustments in
        the update method.

        Parameters
        ----------
        y : TensorVariable
            The observation data at time t.
        Z : TensorVariable
            The design matrix.
        H : TensorVariable
            The observation noise covariance matrix.

        Returns
        -------
        y_masked : TensorVariable
            Observation vector with missing values replaced by zeros.

        Z_masked: TensorVariable
            Design matrix adjusted to exclude the missing states from the information set of observed variables in the
            update step

        H_masked: TensorVariable
            Noise covariance matrix, adjusted to exclude the missing states

        all_nan_flag: float
            1 if the entire state vector is missing

        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        """
        nan_mask = pt.or_(pt.isnan(y), pt.eq(y, self.missing_fill_value))
        all_nan_flag = pt.all(nan_mask).astype(pytensor.config.floatX)
        W = pt.diag(pt.bitwise_not(nan_mask).astype(pytensor.config.floatX))

        Z_masked = W.dot(Z)
        H_masked = W.dot(H)
        y_masked = pt.set_subtensor(y[nan_mask], 0.0)

        return y_masked, Z_masked, H_masked, all_nan_flag

    @staticmethod
    def predict(a, P, c, T, R, Q) -> Tuple[TensorVariable, TensorVariable]:
        """
        Perform the prediction step of the Kalman filter.

        This function computes the one-step forecast of the hidden states and the covariance matrix of the forecasted
        states, based on the current state estimates and model parameters. For computational stability, the estimated
        covariance matrix is forced to by symmetric by averaging it with its own transpose. The prediction equations
        are:

        .. math::

            \begin{align}
            a_{t+1 | t} &= T_t a_{t | t} \\
            P_{t+1 | t} &= T_t P_{t | t} T_t^T + R_t Q_t R_t^T
            \\end{align}


        Parameters
        ----------
        a : TensorVariable
            The current state vector estimate computed by the update step, a[t | t].
        P : TensorVariable
            The current covariance matrix estimate computed by the update step, P[t | t].
        c : TensorVariable
            The hidden state intercept/bias vector.
        T : TensorVariable
            The state transition matrix.
        R : TensorVariable
            The selection matrix.
        Q : TensorVariable
            The state innovation covariance matrix.

        Returns
        -------
        a_hat : TensorVariable
            One-step forecast of the hidden states
        P_hat : TensorVariable
            Covariance matrix of the forecasted hidden states

        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        """
        a_hat = T.dot(a) + c
        P_hat = quad_form_sym(T, P) + quad_form_sym(R, Q)

        return a_hat, P_hat

    @staticmethod
    def update(
        a, P, y, c, d, Z, H, all_nan_flag
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable]:
        """
        Perform the update step of the Kalman filter.

        This function updates the state vector and covariance matrix estimates based on the current observation data,
        previous predictions, and model parameters. The filtering equations are:

        .. math::

            \begin{align}
            \\hat{y}_t &= Z_t a_{t | t-1} \\
            v_t &= y_t - \\hat{y}_t \\
            F_t &= Z_t P_{t | t-1} Z_t^T + H_t \\
            a_{t|t} &= a_{t | t-1} + P_{t | t-1} Z_t^T F_t^{-1} v_t \\
            P_{t|t} &= P_{t | t-1} - P_{t | t-1} Z_t^T F_t^{-1} Z_t P_{t | t-1}
            \\end{align}


        Parameters
        ----------
        a : TensorVariable
            The current state vector estimate, conditioned on information up to time t-1.
        P : TensorVariable
            The current covariance matrix estimate, conditioned on information up to time t-1.
        y : TensorVariable
            The observation data at time t.
        c : TensorVariable
            The matrix c.
        d : TensorVariable
            The matrix d.
        Z : TensorVariable
            The matrix Z.
        H : TensorVariable
            The matrix H.
        all_nan_flag : TensorVariable
            A binary flag tensor indicating whether there are any missing values in the observation data.

        Returns
        -------
        Tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable]
            A tuple containing the updated state vector `a_filtered`, the updated covariance matrix `P_filtered`, the
            predicted observation `obs_mu`, the predicted observation covariance matrix `obs_cov`, and the log-likelihood `ll`.
        """
        raise NotImplementedError

    def kalman_step(self, *args) -> Tuple:
        """
        Performs a single iteration of the Kalman filter, which is composed of two steps : an update step and a
        prediction step. The timing convention follows [1], in which initial state and covariance estimates a0 and P0
        are taken to be predictions. As a result, the update step is applied first. The update step computes:

        .. math::

            \begin{align}
            \\hat{y}_t &= Z_t a_{t | t-1} \\
            v_t &= y_t - \\hat{y}_t \\
            F_t &= Z_t P_{t | t-1} Z_t^T + H_t \\
            a_{t|t} &= a_{t | t-1} + P_{t | t-1} Z_t^T F_t^{-1} v_t \\
            P_{t|t} &= P_{t | t-1} - P_{t | t-1} Z_t^T F_t^{-1} Z_t P_{t | t-1}
            \\end{align}

        Where the quantities :math:`a_{t|t}` and :math:`P_{t|t}` are the best linear estimates of the hidden states
        at time t, incorporating all information up to and including the observation :math:`y_t`. After the update step,
        new one-step forecasts of the hidden states can be obtained by applying the model transition dynamics in
        the prediction step:

        .. math::

            \begin{align}
            a_{t+1 | t} &= T_t a_{t | t} \\
            P_{t+1 | t} &= T_t P_{t | t} T_t^T + R_t Q_t R_t^T
            \\end{align}

        Recursive application of these two steps results in the best linear estimate of the hidden states, including
        missing values and observations subject to measurement error.

        Parameters
        ----------
        Kalman filter inputs:
            y, a, P, c, d, T, Z, R, H, Q. See the docstring for the kalman filter class for details.

        Returns
        ----------
        a_filtered : TensorVariable
            Best linear estimate of hidden states given all information up to and including the present
             observation, a[t | t].

        a_hat: TensorVariable
            One-step forecast of next-period hidden states given all information up to and including the present
            observation, a[t+1 | t]

        obs_mu: TensorVariable
            Estimates of the current observation given all information available prior to the current state,
             d + Z @ a[t | t-1]

        P_filtered: TensorVariable
            Best linear estimate of the covariance between hidden states, given all information up to and including
            the present observation, P[t | t]

        P_hat: TensorVariable
            Covariance between the one-step forecasted hidden states given all information up to and including the
            present observation, P[t+1 | t]

        obs_cov: TensorVariable
            Covariance between estimated present observations, given all information available prior to the current
            state, Z @ P[t | t-1] @ Z.T + H

        ll: float
            Likelihood of the time t observation vector under the multivariate normal distribution parameterized by
            `obs_mu` and `obs_cov`

        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        """
        y, a, P, c, d, T, Z, R, H, Q = self.unpack_args(args)
        y_masked, Z_masked, H_masked, all_nan_flag = self.handle_missing_values(y, Z, H)

        a_filtered, P_filtered, obs_mu, obs_cov, ll = self.update(
            y=y_masked, a=a, c=c, d=d, P=P, Z=Z_masked, H=H_masked, all_nan_flag=all_nan_flag
        )

        P_filtered = stabilize(P_filtered, self.cov_jitter)

        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, c=c, T=T, R=R, Q=Q)
        outputs = (a_filtered, a_hat, obs_mu, P_filtered, P_hat, obs_cov, ll)

        return outputs


class StandardFilter(BaseFilter):
    """
    Basic Kalman Filter
    """

    def update(self, a, P, y, c, d, Z, H, all_nan_flag):
        """
        Compute one-step forecasts for observed states conditioned on information up to, but not including, the current
        timestep, `y_hat`, along with the forcast covariance matrix, `F`. Marginalize over observed states to obtain
        the best linear estimate of the unobserved states, `a_filtered`, as well as the associated covariance matrix,
        `P_filtered`,  conditioned on all information, up to and including the present.

        Derivation of the Kalman filter, along with a deeper discussion of the computational elements, can be found in
        [1].

        Parameters
        ----------
        a : TensorVariable
            The current state vector estimate, conditioned on information up to time t-1.

        P : TensorVariable
            The current covariance matrix estimate, conditioned on information up to time t-1.

        y : TensorVariable
            Observations at time t.

        c : TensorVariable
            Latent state bias term.

        d : TensorVariable
            Observed state bias term.

        Z : TensorVariable
            Linear map between unobserved and observed states.

        H : TensorVariable
            Observation noise covariance matrix

        all_nan_flag : TensorVariable
            A flag indicating whether all elements in the data `y` are NaNs.

        Returns
        -------
        Tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable, float]
            A tuple containing the updated state vector `a_filtered`, the updated covariance matrix `P_filtered`,
            the one-step forecast mean `y_hat`, one-step forcast covariance matrix  `F`, and the log-likelihood of
            the data, given the one-step forecasts, `ll`.

        References
        ----------
        .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
               2nd ed, Oxford University Press, 2012.
        """
        y_hat = d + Z.dot(a)
        v = y - y_hat

        PZT = P.dot(Z.T)
        F = Z.dot(PZT) + stabilize(H, self.cov_jitter)

        F_inv = pt.linalg.solve(F, self.eye_endog, assume_a="pos", check_finite=False)

        K = PZT.dot(F_inv)
        I_KZ = self.eye_states - K.dot(Z)

        a_filtered = a + K.dot(v)
        P_filtered = quad_form_sym(I_KZ, P) + quad_form_sym(K, H)

        inner_term = matrix_dot(v.T, F_inv, v)
        F_logdet = pt.log(pt.linalg.det(F))

        ll = pt.switch(
            all_nan_flag,
            0.0,
            -0.5 * (MVN_CONST + F_logdet + inner_term).ravel()[0],
        )

        return a_filtered, P_filtered, y_hat, F, ll


class CholeskyFilter(BaseFilter):
    """ "
    Kalman filter with Cholesky factorization

    Kalman filter implementation using a Cholesky factorization plus pt.solve_triangular to (attempt) to speed up
    inversion of the observation covariance matrix `F`.

    """

    # TODO: Can the entire Kalman filter process be re-written, starting from P0_chol, so it's not necessary to compute
    #     cholesky(F) at every iteration?

    def update(self, a, P, y, c, d, Z, H, all_nan_flag):
        y_hat = Z.dot(a) + d
        v = y - y_hat

        PZT = P.dot(Z.T)

        # If everything is missing, F will be [[0]] and F_chol will raise an error, so add identity to avoid the error
        F = Z.dot(PZT) + stabilize(H, self.cov_jitter)
        F_chol = pt.linalg.cholesky(F)

        # If everything is missing, K = 0, IKZ = I
        K = solve_triangular(F_chol.T, solve_triangular(F_chol, PZT.T)).T
        I_KZ = self.eye_states - K.dot(Z)

        a_filtered = a + K.dot(v)
        P_filtered = quad_form_sym(I_KZ, P) + quad_form_sym(K, H)

        inner_term = solve_triangular(F_chol.T, solve_triangular(F_chol, v))
        n = y.shape[0]

        ll = pt.switch(
            all_nan_flag,
            0.0,
            (
                -0.5 * (n * MVN_CONST + (v.T @ inner_term).ravel()) - pt.log(pt.diag(F_chol)).sum()
            ).ravel()[0],
        )

        return a_filtered, P_filtered, y_hat, F, ll


class SingleTimeseriesFilter(BaseFilter):
    """
    Kalman filter optimized for univariate timeseries

    If there is only a single observed timeseries, regardless of the number of hidden states, there is no need to
    perform a matrix inversion anywhere in the filter.
    """

    # TODO: This class should eventually be made irrelevant by pytensor re-writes.
    def check_params(self, data, a0, P0, c, d, T, Z, R, H, Q):
        """ "
        Wrap the data in an `Assert` `Op` to ensure there is only one observed state.
        """
        data = assert_data_is_1d(data, pt.eq(data.shape[1], 1))

        return data, a0, P0, c, d, T, Z, R, H, Q

    def update(self, a, P, y, c, d, Z, H, all_nan_flag):
        y_hat = d + Z.dot(a)
        v = y - y_hat.ravel()

        PZT = P.dot(Z.T)

        # F is scalar, K is a column vector
        F = stabilize(Z.dot(PZT) + H, self.cov_jitter).ravel()

        K = PZT / F
        I_KZ = self.eye_states - K.dot(Z)

        a_filtered = a + (K * v).ravel()

        P_filtered = quad_form_sym(I_KZ, P) + quad_form_sym(K, H)

        ll = pt.switch(all_nan_flag, 0.0, -0.5 * (MVN_CONST + pt.log(F) + v**2 / F)).ravel()[0]

        return a_filtered, P_filtered, pt.atleast_1d(y_hat), pt.atleast_2d(F), ll


class SteadyStateFilter(BaseFilter):
    """
    Kalman Filter using Steady State Covariance

    This filter avoids the need to invert the covariance matrix of innovations at each time step by solving the
    Discrete Algebraic Riccati Equation associated with the filtering problem once and for all at initialization and
    uses the resulting steady-state covariance matrix in each step.

    The innovation covariance matrix will always converge to the steady state value as T -> oo, so this filter will
    only have differences from the standard approach in the early steps (T < 10?). A process of "learning" is lost.
    """

    def build_graph(
        self,
        data,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        mode=None,
        return_updates=False,
        missing_fill_value=None,
        cov_jitter=None,
    ) -> List[TensorVariable]:
        """
        Need to override the base step to add an argument to self.update, passing F_inv at every step.
        """
        if missing_fill_value is None:
            missing_fill_value = MISSING_FILL
        if cov_jitter is None:
            cov_jitter = JITTER_DEFAULT

        self.mode = mode
        self.missing_fill_value = missing_fill_value
        self.cov_jitter = cov_jitter
        self.initialize_eyes(R, Z)

        data, a0, P0, *params = self.check_params(data, a0, P0, c, d, T, Z, R, H, Q)
        sequences, non_sequences, seq_names, non_seq_names = split_vars_into_seq_and_nonseq(
            params, PARAM_NAMES
        )
        self.seq_names = seq_names
        self.non_seq_names = non_seq_names
        c, d, T, Z, R, H, Q = params

        if len(sequences) > 0:
            assert ValueError(
                "All system matrices must be time-invariant to use the SteadyStateFilter"
            )

        P_steady = solve_discrete_are(T.T, Z.T, matrix_dot(R, Q, R.T), H)
        F = matrix_dot(Z, P_steady, Z.T) + H
        F_inv = pt.linalg.solve(F, pt.eye(F.shape[0]), assume_a="pos", check_finite=False)

        results, updates = pytensor.scan(
            self.kalman_step,
            sequences=[data],
            outputs_info=[None, a0, None, None, P_steady, None, None],
            non_sequences=[c, d, F_inv, T, Z, R, H, Q],
            name="forward_kalman_pass",
            mode=get_mode(self.mode),
        )

        return self._postprocess_scan_results(results, a0, P0, n=data.shape[0])

    def update(self, a, P, c, d, F_inv, y, Z, H, all_nan_flag):
        y_hat = Z.dot(a) + d
        v = y - y_hat

        PZT = P.dot(Z.T)

        F = Z.dot(PZT) + stabilize(H, self.cov_jitter)
        K = PZT.dot(F_inv)

        I_KZ = self.eye_states - K.dot(Z)

        a_filtered = a + K.dot(v)
        P_filtered = quad_form_sym(I_KZ, P) + quad_form_sym(K, H)

        inner_term = matrix_dot(v.T, F_inv, v)
        ll = pt.switch(
            all_nan_flag,
            0.0,
            -0.5 * (MVN_CONST + pt.log(pt.linalg.det(F)) + inner_term).ravel()[0],
        )

        return a_filtered, P_filtered, y_hat, F, ll

    def kalman_step(self, y, a, P, c, d, F_inv, T, Z, R, H, Q):
        """
        Need to override the base step to add an argument to self.update, passing F_inv at every step.
        """

        y_masked, Z_masked, H_masked, all_nan_flag = self.handle_missing_values(y, Z, H)
        a_filtered, P_filtered, obs_mu, obs_cov, ll = self.update(
            y=y_masked,
            a=a,
            P=P,
            c=c,
            d=d,
            F_inv=F_inv,
            Z=Z_masked,
            H=H_masked,
            all_nan_flag=all_nan_flag,
        )

        P_filtered = stabilize(P_filtered, self.cov_jitter)
        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, c=c, T=T, R=R, Q=Q)

        return a_filtered, a_hat, obs_mu, P_filtered, P_hat, obs_cov, ll


class UnivariateFilter(BaseFilter):
    """
    The univariate kalman filter, described in [1], section 6.4.2, avoids inversion of the F matrix, as well as two
    matrix multiplications, at the cost of an additional loop. Note that the name doesn't mean there's only one
    observed time series, that's the SingleTimeSeries filter. This is called univariate because it updates the state
    mean and covariance matrices one variable at a time, using an inner-inner loop.

    This is useful when states are perfectly observed, because the F matrix can easily become degenerate in these cases.

    References
    ----------
    .. [1] Durbin, J., and S. J. Koopman. Time Series Analysis by State Space Methods.
            2nd ed, Oxford University Press, 2012.

    """

    def _univariate_inner_filter_step(self, y, Z_row, d_row, sigma_H, nan_flag, a, P):
        y_hat = Z_row.dot(a) + d_row
        v = y - y_hat

        PZT = P.dot(Z_row.T)
        F = Z_row.dot(PZT) + sigma_H

        # Set the zero flag for F first, then jitter it to avoid a divide-by-zero NaN later
        F_zero_flag = pt.or_(pt.eq(F, 0), nan_flag)
        F = F + self.cov_jitter

        # If F is zero (implies y is NAN or another degenerate case), then we want:
        # K = 0, a = a, P = P, ll = 0
        K = PZT / F * (1 - F_zero_flag)

        a_filtered = a + K * v
        P_filtered = P - pt.outer(K, K) * F

        ll_inner = pt.switch(F_zero_flag, 0.0, pt.log(F) + v**2 / F)

        return a_filtered, P_filtered, pt.atleast_1d(y_hat), pt.atleast_2d(F), ll_inner

    def kalman_step(self, y, a, P, c, d, T, Z, R, H, Q):
        nan_mask = pt.isnan(y)

        W = pt.set_subtensor(pt.eye(y.shape[0])[nan_mask, nan_mask], 0.0)
        Z_masked = W.dot(Z)
        H_masked = W.dot(H)
        y_masked = pt.set_subtensor(y[nan_mask], 0.0)

        result, updates = pytensor.scan(
            self._univariate_inner_filter_step,
            sequences=[y_masked, Z_masked, d, pt.diag(H_masked), nan_mask],
            outputs_info=[a, P, None, None, None],
            mode=get_mode(self.mode),
            name="univariate_inner_scan",
        )

        a_filtered, P_filtered, obs_mu, obs_cov, ll_inner = result
        a_filtered, P_filtered, obs_mu, obs_cov = (
            a_filtered[-1],
            P_filtered[-1],
            obs_mu[-1],
            obs_cov[-1],
        )

        P_filtered = stabilize(0.5 * (P_filtered + P_filtered.T), self.cov_jitter)
        a_hat, P_hat = self.predict(a=a_filtered, P=P_filtered, c=c, T=T, R=R, Q=Q)

        ll = -0.5 * ((pt.neq(ll_inner, 0).sum()) * MVN_CONST + ll_inner.sum())

        return a_filtered, a_hat, obs_mu, P_filtered, P_hat, obs_cov, ll
