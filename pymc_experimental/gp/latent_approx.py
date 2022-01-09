import numpy as np
import aesara.tensor as at
import pymc as pm

from pymc.gp.util import (
    JITTER_DEFAULT,
    cholesky,
    conditioned_vars,
    infer_size,
    replace_with_values,
    solve_lower,
    solve_upper,
    stabilize,
)


class ProjectedProcess(pm.gp.Latent):
    def __init__(self, n_inducing, *, mean_func=pm.gp.mean.Zero(), cov_func=pm.gp.cov.Constant(0.0)):
        self.n_inducing = n_inducing
        super().__init__(mean_func=mean_func, cov_func=cov_func)
        
    def _build_prior(self, name, X, Xu, jitter=JITTER_DEFAULT, **kwargs):
        mu = self.mean_func(X)
        Kuu = self.cov_func(Xu)
        L = cholesky(stabilize(Kuu, jitter))
        
        n_inducing_points = infer_size(Xu, kwargs.pop("size", None))
        v = pm.Normal(name + "_u_rotated_", mu=0.0, sigma=1.0, size=n_inducing_points, **kwargs)
        u = pm.Deterministic(name + "_u", L @ v)
        
        Kfu = self.cov_func(X, Xu)
        Kuuiu = solve_upper(at.transpose(L), solve_lower(L, u))

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
        tmp = solve_lower(L, at.transpose(Ksu))
        Qss = at.transpose(tmp) @ tmp  #Qss = tt.dot(tt.dot(Ksu, tt.nlinalg.pinv(Kuu)), Ksu.T) 
        Kss = self.cov_func(Xnew)
        Lss = cholesky(stabilize(Kss - Qss, jitter))    
        return mu, Lss
    
    def conditional(self, name, Xnew, jitter, **kwargs):
        mu, chol = self._build_conditional(name, Xnew, self.Xu, self.L, self.Kuuiu, jitter, **kwargs)
        return pm.MvNormal("fnew", mu=mu, chol=chol)


class HSGP(pm.gp.Latent):
    ## inputs, M, c

    def __init__(self, n_basis, c=3/2, *, mean_func=pm.gp.mean.Zero(), cov_func=pm.gp.cov.Constant(0.0)):
        ## TODO: specify either c or L
        self.M = n_basis
        self.c = c
        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def _validate_cov_func(self, cov_func):
        ## TODO: actually validate it.  Right now this fails unless cov func is exactly 
        # in the form eta**2 * pm.gp.cov.Matern12(...) and will error otherwise.
        cov, scaling_factor = cov_func.factor_list
        return scaling_factor, cov.ls, cov.spectral_density

    def prior(self, name, X, **kwargs):
        f, Phi, L, spd, beta, Xmu, Xsd = self._build_prior(name, X, **kwargs)
        self.X, self.f = X, f
        self.Phi, self.L, self.spd, self.beta = Phi, L, spd, beta
        self.Xmu, self.Xsd = Xmu, Xsd
        return f

    def _generate_basis(self, X, L):
        indices = at.arange(1, self.M + 1)
        m1 = (np.pi / (2.0 * L)) * at.tile(L + X, self.M)
        m2 = at.diag(indices)
        Phi = at.sin(m1 @ m2) / at.sqrt(L)
        omega = (np.pi * indices) / (2.0 * L)
        return Phi, omega

    def _build_prior(self, name, X, **kwargs):
        n_obs = pm.gp.util.infer_size(X, kwargs.get("n_obs"))

        # standardize input
        X = at.as_tensor_variable(X)
        Xmu = at.mean(X, axis=0)
        Xsd = at.std(X, axis=0)
        Xz = (X - Xmu) / Xsd

        # define L using Xz and c
        La = at.abs(at.min(Xz)).eval()
        Lb = at.max(Xz)
        L = self.c * at.max([La, Lb])

        # make basis and omega, spectral density
        Phi, omega = self._generate_basis(Xz, L)
        scale, ls, spectral_density = self._validate_cov_func(self.cov_func)
        spd = scale * spectral_density(omega, ls / Xsd).flatten()

        beta = pm.Normal(f'{name}_coeffs_', size=self.M)
        f = pm.Deterministic(name, self.mean_func(X) + at.dot(Phi * at.sqrt(spd), beta))
        return f, Phi, L, spd, beta, Xmu, Xsd

    def _build_conditional(self, Xnew, Xmu, Xsd, L, beta):
        Xnewz = (Xnew - Xmu) / Xsd
        Phi, omega = self._generate_basis(Xnewz, L)
        scale, ls, spectral_density = self._validate_cov_func(self.cov_func)
        spd = scale * spectral_density(omega, ls / Xsd).flatten()
        return self.mean_func(Xnew) + at.dot(Phi * at.sqrt(spd), beta)

    def conditional(self, name, Xnew):
        # warn about extrapolation
        fnew = self._build_conditional(Xnew, self.Xmu, self.Xsd, self.L, self.beta)
        return pm.Deterministic(name, fnew)


class ExpQuad(pm.gp.cov.ExpQuad):
    @staticmethod
    def spectral_density(omega, ls):
        # univariate spectral denisty, implement multi
        return at.sqrt(2 * np.pi) * ls * at.exp(-0.5 * ls**2 * omega**2)

class Matern52(pm.gp.cov.Matern52):
    @staticmethod
    def spectral_density(omega, ls):
        # univariate spectral denisty, implement multi
        # https://arxiv.org/pdf/1611.06740.pdf
        lam = at.sqrt(5) * (1.0 / ls)
        return (16.0 / 3.0) * lam**5 * (1.0 / (lam**2 + omega**2)**3)

class Matern32(pm.gp.cov.Matern32):
    @staticmethod
    def spectral_density(omega, ls):
        # univariate spectral denisty, implement multi
        # https://arxiv.org/pdf/1611.06740.pdf
        lam = np.sqrt(3.0) * (1.0 / ls)
        return 4.0 * lam**3 * (1.0 / at.square(lam**2 + omega**2))

class Matern12(pm.gp.cov.Matern12):
    @staticmethod
    def spectral_density(omega, ls):
        # univariate spectral denisty, implement multi
        # https://arxiv.org/pdf/1611.06740.pdf
        lam = 1.0 / ls
        return 2.0 * lam * (1.0 / (lam**2 + omega**2))



