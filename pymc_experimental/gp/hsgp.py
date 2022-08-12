import aesara.tensor as at
import numpy as np
import pymc as pm


class HSGP(pm.gp.gp.Base):
    def __init__(
        self,
        n_basis,
        c=3 / 2,
        L=None,
        *,
        mean_func=pm.gp.mean.Zero(),
        cov_func=pm.gp.cov.Constant(0.0),
    ):
        arg_err_msg = (
            "`n_basis` and L, if provided, must be lists or tuples, with one element per active "
            "dimension."
        )
        try:
            if len(n_basis) != cov_func.D:
                raise ValueError(arg_err_msg)
        except TypeError as e:
            raise ValueError(arg_err_msg) from e

        if L is not None and len(L) != cov_func.D:
            raise ValueError(arg_err_msg)

        self.M = n_basis
        self.L = L
        self.c = c
        self.D = cov_func.D

        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def __add__(self, other):
        raise NotImplementedError("Additive HSGP's isn't supported ")

    def _set_boundary(self, X):
        if self.L is None:
            # Define new L based on c and X range
            La = at.abs(at.min(X, axis=0))
            Lb = at.abs(at.max(X, axis=0))
            self.L = self.c * at.max(at.stack((La, Lb)), axis=0)
        else:
            self.L = at.as_tensor_variable(self.L)

    @staticmethod
    def _eigendecomposition(X, L, M, D):
        """Construct the eigenvalues and eigenfunctions of the Laplace operator."""
        m_star = at.prod(M)
        S = np.meshgrid(*[np.arange(1, 1 + M[d]) for d in range(D)])
        S = np.vstack([s.flatten() for s in S]).T
        eigvals = at.square((np.pi * S) / (2 * L))
        phi = at.ones((X.shape[0], m_star))
        for d in range(D):
            c = 1.0 / np.sqrt(L[d])
            phi *= c * at.sin(at.sqrt(eigvals[:, d]) * (at.tile(X[:, d][:, None], m_star) + L[d]))
        omega = at.sqrt(eigvals)
        return omega, phi, m_star

    def approx_K(self, X):
        # construct eigenvectors after slicing X
        X, _ = self.cov_func._slice(X)
        omega, phi, _ = self._eigendecomposition(X, self.L, self.M, self.D)
        psd = self.cov_func.psd(omega)
        return at.dot(phi * psd, at.transpose(phi))

    def prior(self, name, X, **kwargs):
        X, _ = self.cov_func._slice(X)
        self._set_boundary(X)
        omega, phi, m_star = self._eigendecomposition(X, self.L, self.M, self.D)
        psd = self.cov_func.psd(omega)
        self.beta = pm.Normal(f"{name}_coeffs_", size=m_star)
        self.f = pm.Deterministic(
            name, self.mean_func(X) + at.squeeze(at.dot(phi, self.beta * psd))
        )
        return self.f

    def _build_conditional(self, name, Xnew):
        Xnew, _ = self.cov_func._slice(Xnew)
        omega, phi, _ = self._eigendecomposition(Xnew, self.L, self.M, self.D)
        psd = self.cov_func.psd(omega)
        return self.mean_func(Xnew) + at.squeeze(at.dot(phi, self.beta * psd))

    def conditional(self, name, Xnew):
        fnew = self._build_conditional(name, Xnew)
        return pm.Deterministic(name, fnew)
