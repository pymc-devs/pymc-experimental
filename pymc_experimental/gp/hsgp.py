import pymc as pm
import aesara.tensor as at


class HSGP(Latent):
    ## inputs, M, c

    def __init__(self, M, c=3/2, *, mean_func=pm.gp.mean.Zero(), cov_func=pm.gp.cov.Constant(0.0)):
        ## TODO: specify either c or L
        self.M = M
        self.c = c
        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def _validate_cov_func(self, cov_func):
        ## TODO: actually validate it
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
        Xmu, Xsd = self.Xmu, self.Xsd
        L, beta = self.L, self.beta
        fnew = self._build_conditional(Xnew, Xmu, Xsd, L, beta)
        return pm.Deterministic(name, fnew)



# ExpQuad
#staticmethod
def spectral_density(omega, ls):
    # univariate spectral denisty, implement multi
    return at.sqrt(2 * np.pi) * ls * at.exp(-0.5 * ls**2 * omega**2)

# Matern52
#staticmethod
def spectral_density(omega, ls):
    # univariate spectral denisty, implement multi
    # https://arxiv.org/pdf/1611.06740.pdf
    lam = at.sqrt(5) * (1.0 / ls)
    return (16.0 / 3.0) * lam**5 * (1.0 / (lam**2 + omega**2)**3)

# Matern32
#staticmethod
def spectral_density(omega, ls):
    # univariate spectral denisty, implement multi
    # https://arxiv.org/pdf/1611.06740.pdf
    lam = np.sqrt(3.0) * (1.0 / ls)
    return 4.0 * lam**3 * (1.0 / at.square(lam**2 + omega**2))

# Matern12
#@staticmethod
def spectral_density(omega, ls):
    # univariate spectral denisty, implement multi
    # https://arxiv.org/pdf/1611.06740.pdf
    lam = 1.0 / ls
    return 2.0 * lam * (1.0 / (lam**2 + omega**2))



