import numpy as np
import aesara.tensor as at
import pymc as pm

from pymc.gp.util import (
    JITTER_DEFAULT,
    cholesky,
    conditioned_vars,
    replace_with_values,
)


class ExpQuad(pm.gp.cov.ExpQuad):
    def psd(self, omega):
        D = len(self.active_dims)
        ls = at.ones(D) * self.ls
        c = at.power(at.sqrt(2.0 * np.pi), D) * at.prod(ls)
        return c * at.exp(-0.5 * at.dot(omega, ls))
    

class Matern52(pm.gp.cov.Matern52):
    def psd(self, omega):
        D = len(self.active_dims)
        ls = at.ones(D) * self.ls
        D52 = (D + 5) / 2
        num = at.power(2, D) * at.power(np.pi, D / 2) * at.gamma(D52) * at.power(5, 5 / 2)
        den = 0.75 * at.sqrt(np.pi) * at.power(ls, 5)
        return (num / den) * at.power(5.0 + at.dot(omega, ls), D52)
    

class HSGP(pm.gp.gp.Base):
    def __init__(self, M, c=3/2, L=None, *, mean_func=pm.gp.mean.Zero(), cov_func=pm.gp.cov.Constant(0.0)):
        self.M = M
        self.c = c
        self.L = L
        self.D = len(cov_func.active_dims)
        self.m_star = at.power(self.M, self.D)
        super().__init__(mean_func=mean_func, cov_func=cov_func)
        
    def _evaluate_psd(self, omega):
        
        # Unpack and validate covariance function
        #cov, scale = self.cov_func.factor_list
        
        return self.cov_func.psd(omega)
        
    @staticmethod
    def _standardize(self, X):
        Xmu = at.mean(X[:, self.cov_func.active_dims], axis=0)
        Xsd = at.std(X[:, self.cov_func.active_dims], axis=0)
        Xz = (X[:, self.cov_func.active_dims] - Xmu) / Xsd
        return Xz, Xmu, Xsd
    
    def _construct_basis(self, X): 
        """ Construct the set of basis vectors and associated eigenvalue array.
        """
        S = np.meshgrid(*[np.arange(1, 1 + self.M) for _ in range(self.D)])
        S = np.vstack([s.flatten() for s in S]).T
        eigvals = at.square((np.pi * S) / (2 * self.L))

        phi_shape = (X.shape[0], self.m_star)
        phi = at.ones(phi_shape)
        for d in range(self.D):
            c = 1.0 / np.sqrt(self.L[d])
            phi *= c * at.sin(at.sqrt(eigvals[:, d]) * (at.tile(X[:, d][:, None], self.m_star) + self.L[d]))
        return eigvals, phi 

    def _build_prior(self, name, X, **kwargs):
        X = at.as_tensor_variable(X)
        
        if self.L is None:
            
            # Define new L based on c and X range
            La = at.abs(at.min(X, axis=0))
            Lb = at.abs(at.max(X, axis=0))
            self.L = self.c * at.max(at.stack((La, Lb)), axis=0)
            
        else:
            # If L is passed as a scalar, put it into a one-element list.
            if np.isscalar(self.L):
                self.L = [self.L]
               
            if len(self.L) != self.D:
                raise ValueError(
                    (
                        "Must provide one L for each active dimension.  `len(L)` must ",
                         "equal the number of active dimensions of your covariance function."
                    )
                )
           
            ## If L is provided, don't rescale X
            self.L = at.as_tensor_variable(self.L)
           
        # Construct basis and eigenvalues
        eigvals, phi = self._construct_basis(X)
        omega = at.sqrt(eigvals)
        psd = self._evaluate_psd(omega)
        beta = pm.Normal(f'{name}_coeffs_', size=self.m_star)
        f = pm.Deterministic(name, self.mean_func(X) + at.squeeze(at.dot(phi * beta, psd)))   
        return eigvals, phi, psd, f, beta
    
    @property
    def basis(self):
        try:
            return self.phi
        except AttributeError:
            raise RuntimeError("Must construct the prior first by calling `.prior.") 
   
    @property
    def power_spectral_density(self):
        try:
            return self.psd
        except AttributeError:
            raise RuntimeError("Must construct the prior first by calling `.prior.") 
    
    @property
    def omega(self):
        try:
            return at.sqrt(self.eigvals)
        except AttributeError:
            raise RuntimeError("Must construct the prior first by calling `.prior.") 

    def prior(self, name, X, **kwargs):
        self.eigvals, self.phi, self.psd, self.f, self.beta = self._build_prior(name, X, **kwargs)
        return self.f
    
    def _build_conditional(self, name, Xnew):
        eigvals, phi = self._construct_basis(Xnew)
        omega = at.sqrt(eigvals)
        psd = self._evaluate_psd(omega)
        return self.mean_func(Xnew) + at.squeeze(at.dot(phi * self.beta, psd))
       
    def conditional(self, name, Xnew):
        fnew = self._build_conditional(name, Xnew)
        return pm.Deterministic(name, fnew)