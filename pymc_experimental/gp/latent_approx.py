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
from functools import partial

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.gp.util import JITTER_DEFAULT, stabilize
from pytensor.tensor.linalg import cholesky, solve_triangular

solve_lower = partial(solve_triangular, lower=True)
solve_upper = partial(solve_triangular, lower=False)


class LatentApprox(pm.gp.Latent):
    ## TODO: use strings to select approximation, like pm.gp.MarginalApprox?
    pass


class ProjectedProcess(pm.gp.Latent):
    ## AKA: DTC
    def __init__(
        self, n_inducing, *, mean_func=pm.gp.mean.Zero(), cov_func=pm.gp.cov.Constant(0.0)
    ):
        self.n_inducing = n_inducing
        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def _build_prior(self, name, X, Xu, jitter=JITTER_DEFAULT, **kwargs):
        mu = self.mean_func(X)
        Kuu = self.cov_func(Xu)
        L = cholesky(stabilize(Kuu, jitter))

        n_inducing_points = np.shape(Xu)[0]
        v = pm.Normal(name + "_u_rotated_", mu=0.0, sigma=1.0, size=n_inducing_points, **kwargs)
        u = pm.Deterministic(name + "_u", L @ v)

        Kfu = self.cov_func(X, Xu)
        Kuuiu = solve_upper(pt.transpose(L), solve_lower(L, u))

        return pm.Deterministic(name, mu + Kfu @ Kuuiu), Kuuiu, L

    def prior(self, name, X, Xu=None, jitter=JITTER_DEFAULT, **kwargs):
        if Xu is None and self.n_inducing is None:
            raise ValueError
        elif Xu is None:
            if isinstance(X, np.ndarray):
                Xu = pm.gp.util.kmeans_inducing_points(self.n_inducing, X, **kwargs)

        f, Kuuiu, L = self._build_prior(name, X, Xu, jitter, **kwargs)
        self.X, self.Xu = X, Xu
        self.L, self.Kuuiu = L, Kuuiu
        self.f = f
        return f

    def _build_conditional(self, name, Xnew, Xu, L, Kuuiu, jitter, **kwargs):
        Ksu = self.cov_func(Xnew, Xu)
        mu = self.mean_func(Xnew) + Ksu @ Kuuiu
        tmp = solve_lower(L, pt.transpose(Ksu))
        Qss = pt.transpose(tmp) @ tmp  # Qss = tt.dot(tt.dot(Ksu, tt.nlinalg.pinv(Kuu)), Ksu.T)
        Kss = self.cov_func(Xnew)
        Lss = cholesky(stabilize(Kss - Qss, jitter))
        return mu, Lss

    def conditional(self, name, Xnew, jitter=1e-6, **kwargs):
        mu, chol = self._build_conditional(
            name, Xnew, self.Xu, self.L, self.Kuuiu, jitter, **kwargs
        )
        return pm.MvNormal(name, mu=mu, chol=chol)


class KarhunenLoeveExpansion(pm.gp.Latent):
    def __init__(
        self,
        variance_limit=None,
        n_eigs=None,
        *,
        mean_func=pm.gp.mean.Zero(),
        cov_func=pm.gp.cov.Constant(0.0),
    ):
        self.variance_limit = variance_limit
        self.n_eigs = n_eigs
        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def _build_prior(self, name, X, jitter=1e-6, **kwargs):
        mu = self.mean_func(X)
        Kxx = pm.gp.util.stabilize(self.cov_func(X), jitter)
        vals, vecs = pt.linalg.eigh(Kxx)
        ## NOTE: REMOVED PRECISION CUTOFF
        if self.variance_limit is None:
            n_eigs = self.n_eigs
        else:
            if self.variance_limit == 1:
                n_eigs = len(vals)
            else:
                n_eigs = ((vals[::-1].cumsum() / vals.sum()) > self.variance_limit).nonzero()[0][0]
        U = vecs[:, -n_eigs:]
        s = vals[-n_eigs:]
        basis = U * pt.sqrt(s)

        coefs_raw = pm.Normal(f"_gp_{name}_coefs", mu=0, sigma=1, size=n_eigs)
        # weight = pm.HalfNormal(f"_gp_{name}_sd")
        # coefs = weight * coefs_raw # dont understand this prior, why weight * coeffs_raw?
        f = basis @ coefs_raw
        return f, U, s, n_eigs

    def prior(self, name, X, jitter=1e-6, **kwargs):
        f, U, s, n_eigs = self._build_prior(name, X, jitter, **kwargs)
        self.U, self.s, self.n_eigs = U, s, n_eigs
        self.X = X
        self.f = f
        return pm.Deterministic(name, f)

    def _build_conditional(self, Xnew, X, f, U, s, jitter):
        Kxs = self.cov_func(X, Xnew)
        Kss = self.cov_func(Xnew)
        Kxxpinv = U @ pt.diag(1.0 / s) @ U.T
        mus = Kxs.T @ Kxxpinv @ f
        K = Kss - Kxs.T @ Kxxpinv @ Kxs
        L = pm.gp.util.cholesky(pm.gp.util.stabilize(K, jitter))
        return mus, L

    def conditional(self, name, Xnew, jitter=1e-6, **kwargs):
        X, f = self.X, self.f
        U, s = self.U, self.s
        mu, L = self._build_conditional(Xnew, X, f, U, s, jitter)
        return pm.MvNormal(name, mu=mu, chol=L, **kwargs)
