import jax.numpy as jnp


def _bp_int(nu, f, bp):
    wg, bp0, norm = bp
    return (wg * f) @ bp0 / norm


class FactorizedCrossSpectrum:
    """
       C_ell^(nu1xnu2) = A * C_ell * f(nu1) * f(nu2)
    """
    def __init__(self, power, sed):
        self.power = power
        self.sed = sed

    def __call__(self, ell, nus, bps, theta):
        theta_cl = theta[:self.power.n]
        theta_sed = theta[self.power.n:]

        cl = self.power(ell, *theta_cl)

        f_nus = jnp.zeros((len(ell), len(nus)))
        for i, (nu, bp) in enumerate(zip(nus, bps)):
            f_nu = self.sed(nu, *theta_sed)
            f_nus = f_nus.at[:, i].set(_bp_int(nu, f_nu, bp))
        return jnp.einsum("...i,...ij,...il->...ijl", cl, f_nus, f_nus)

    @property
    def n(self):
        return self.power.n + self.sed.n


class FactorizedCrossSpectrumTE:
    """
       C_ell^(nu1xnu2) = A * C_ell * f^T(nu1) * f^E(nu2)
    """
    def __init__(self, power, sedT, sedE):
        self.power = power
        self.sedT = sedT
        self.sedE = sedE

    def __call__(self, ell, nus, bps, theta):
        theta_cl = theta[:self.power.n]
        theta_sedT = theta[self.power.n:self.power.n+self.sedT.n]
        theta_sedE = theta[self.power.n+self.sedT.n:]

        cl = self.power(ell, *theta_cl)

        f_nuT = jnp.zeros((len(ell), len(nus)))
        f_nuE = jnp.zeros((len(ell), len(nus)))

        for i, (nu, bp) in enumerate(zip(nus, bps)):
            f_nu = self.sedT(nu, *theta_sedT)
            f_nuT = f_nuT.at[:, i].set(_bp_int(nu, f_nu, bp))

            f_nu = self.sedE(nu, *theta_sedT)
            f_nuE = f_nuE.at[:, i].set(_bp_int(nu, f_nu, bp))

        return jnp.einsum("...i,...ij,...il->...ijl", cl, f_nuT, f_nuE)

    @property
    def n(self):
        return self.power.n + self.sedT.n + self.sedE.n


class CorrelatedCrossSpectrum:
    """
        For two components, A and B, which are correlated:
        C_ell^(nu1xnu2) =
            a_A * C_ell^A f^A(nu1) f^A(nu2)
            + a_B * C_ell^B f^B(nu1) f^B(nu2)
            - a_(AxB) sqrt(a_A a_B) C_ell^(AxB) (
                  f^A(nu1) f^B(nu2) + f^A(nu2) f^B(nu1)
              )
    """
    def __init__(self, power1, power2, powerx, sed1, sed2):
        self.power1 = power1
        self.power2 = power2
        self.powerx = powerx
        self.sed1 = sed1
        self.sed2 = sed2
        self.n_p1 = (jnp.arange(self.power1.n))
        self.n_p2 = (jnp.arange(self.power2.n) + self.power1.n)
        self.n_px = (jnp.arange(self.powerx.n) + self.power1.n + self.power2.n)
        self.n_s1 = (jnp.arange(self.sed1.n) + self.power1.n + self.power2.n
                     + self.powerx.n)
        self.n_s2 = (jnp.arange(self.sed2.n) + self.power1.n + self.power2.n
                     + self.powerx.n + self.sed1.n)

    def __call__(self, ell, nus, bps, theta):
        theta_cl1 = theta[self.n_p1]
        theta_cl2 = theta[self.n_p2]
        theta_clx = theta[self.n_px]
        theta_s1 = theta[self.n_s1]
        theta_s2 = theta[self.n_s2]

        cl1 = self.power1(ell, *theta_cl1)
        cl2 = self.power2(ell, *theta_cl2)
        clx = self.powerx(ell, *theta_clx)

        f_nus1 = jnp.zeros((len(ell), len(nus)))
        f_nus2 = jnp.zeros((len(ell), len(nus)))
        for i, (nu, bp) in enumerate(zip(nus, bps)):
            f_nu = self.sed1(nu, *theta_s1)
            f_nus1 = f_nus1.at[:, i].set(_bp_int(nu, f_nu, bp))

            f_nu = self.sed2(nu, *theta_s2)
            f_nus2 = f_nus2.at[:, i].set(_bp_int(nu, f_nu, bp))

        comp1 = jnp.einsum("...i,...ij,...il->...ijl", cl1, f_nus1, f_nus1)
        comp2 = jnp.einsum("...i,...ij,...il->...ijl", cl2, f_nus2, f_nus2)
        compx = (clx[:, None, None] * jnp.sqrt(self.power1.amp(*theta_cl1)
                 * self.power2.amp(*theta_cl2))
                 * (f_nus1[:, :, None] * f_nus2[:, None, :]
                    + f_nus1[:, None, :] * f_nus2[:, :, None]))

        return comp1 + comp2 - compx

    @property
    def n(self):
        return (self.power1.n + self.power2.n + self.powerx.n + self.sed1.n
                + self.sed2.n)
