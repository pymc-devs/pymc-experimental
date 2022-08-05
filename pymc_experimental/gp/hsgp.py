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
    def psd(self, omega, ls):
        D = len(self.active_dims)
        ls = at.ones(D) * ls
        c = at.power(at.sqrt(2.0 * np.pi), D) * at.prod(ls)
        return c * at.exp(-0.5 * at.dot(omega, ls))
    

class Matern52(pm.gp.cov.Matern52):
    def psd(self, omega, ls):
        D = len(self.active_dims)
        ls = at.ones(D) * ls
        D52 = (D + 5) / 2
        num = at.power(2, D) * at.power(np.pi, D / 2) * at.gamma(D52) * at.power(5, 5 / 2)
        den = 0.75 * at.sqrt(np.pi) * at.power(ls, 5)
        return (num / den) * at.power(5.0 + at.dot(omega, ls), D52)
    

class HSGP(pm.gp.Base):
    def __init__(self, M, c=3/2, L=None, *, mean_func=pm.gp.mean.Zero(), cov_func=pm.gp.cov.Constant(0.0)):
        self.M = M
        self.c = c
        self.L = L
        self.D = len(cov_func.active_dims)
        super().__init__(mean_func=mean_func, cov_func=cov_func)
        
    def _evaluate_spd(self, cov_func, omega, L, M, Xsd):
        cov, scale = cov_func.factor_list
        return scale * cov.psd(omega, cov.ls / Xsd)
        
    def _standardize(self, X):
        self.Xmu = at.mean(X[:, self.cov_func.active_dims], axis=0)
        self.Xsd = at.std(X[:, self.cov_func.active_dims], axis=0)
        Xz = (X[:, self.cov_func.active_dims] - self.Xmu) / self.Xsd
        return Xz
    
    @staticmethod
    def _construct_basis(X, L, D, M): 
        #         
        S = np.meshgrid(*[np.arange(1, 1 + M) for _ in range(D)])
        S = np.vstack([s.flatten() for s in S]).T
        eigvals = at.square((np.pi * S) / (2 * L))

        m_star = at.power(M, D)
        phi_shape = (X.shape[0], m_star)
        phi = at.ones(phi_shape)
        for d in range(D):
            c = 1.0 / np.sqrt(L[d])
            phi *= c * at.sin(at.sqrt(eigvals[:, d]) * (at.tile(X[:, d][:, None], m_star) + L[d]))
        return eigvals, phi 

    def _build_prior(self, name, X, **kwargs):
        X = at.as_tensor_variable(X)
        
        if self.L is None:
            Xz = self._standardize(X)
            La = at.abs(at.min(Xz, axis=0))
            Lb = at.abs(at.max(Xz, axis=0))
            L = self.c * at.max(at.stack((La, Lb)), axis=0)
            self.L = L
            
        else:
            if np.isscalar(self.L):
                L = [self.L]
                
            if len(L) != self.D:
                raise ValueError("Must provide one L for each active dimension.")
                
            Xz = X
            self.Xmu, self.Xsd = 0.0, 1.0
            L = at.as_tensor_variable(L)
            
        eigvals, phi = self._construct_basis(Xz, L, self.D, self.M)
        
        omega = at.sqrt(eigvals)
        self.omega = omega
        
        spd = self._evaluate_spd(self.cov_func, omega, L, self.M, self.Xsd)
        
        m_star = at.power(self.M, self.D)
        beta = pm.Normal(f'{name}_coeffs_', size=m_star)
        f = pm.Deterministic(name, self.mean_func(X) + at.squeeze(at.dot(phi * beta, spd)))   
        return eigvals, phi, spd, f, beta
    
    def prior(self, name, X, **kwargs):
        eigvals, phi, spd, f, beta = self._build_prior(name, X, **kwargs)
        
        self.eigvals = eigvals
        self.phi = phi
        self.spd = spd
        
        self.f, self.beta = f, beta
        return f
    
    def _build_conditional(self, name, Xnew):
        Xznew = (Xnew[:, self.cov_func.active_dims] - self.Xmu) / self.Xsd
        eigvals, phi = self._construct_basis(Xznew, self.L, self.D, self.M)
        omega = at.sqrt(eigvals)
        spd = self._evaluate_spd(self.cov_func, omega, self.L, self.M, self.Xsd)
        return self.mean_func(Xnew) + at.squeeze(at.dot(phi * self.beta, spd))
       
    def conditional(self, name, Xnew):
        # TODO, error if Xnew outside bounds given by L
        fnew = self._build_conditional(name, Xnew)
        return pm.Deterministic(name, fnew)
