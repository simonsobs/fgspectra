r"""
Jax-compatible cross-spectrum calculation.

This module differs from the one found in cross.py by being more oriented
towards jax frameworks. For example, the factories in cross.py expect
dictionary-like arguments for function calls, whereas the functions in this
module are oriented towards vectorized calls.

The main difference from the cross.py functions is bandpass integration: the
function calls here expect a pre-computed bandpass integration weight. This
pre-computation is less practical for the end-user, but much faster when having
to compute large amounts of gradients through the full model.
"""

from .model import Model

import jax.numpy as jnp


def _bp_int(f, bp):
    wg, bp0, norm = bp
    return (wg * f) @ bp0 / norm


class FactorizedCrossSpectrum(Model):
    r"""Factorized cross-spectrum.

    Cross-spectrum of a single component for which the frequency and multipole
    scaling are independent and factorizable.

    .. math:: xC_\ell^{(ij)} = A * C_\ell * f(\nu_i) * f(\nu_j)
    """

    def __init__(self, sed, cl, **kwargs):
        r"""
        Parameters
        ----------
        sed : callable
            :math:`f(\nu)`. It returns an array with shape ``(..., freq)``.
            It can be :class:`fgspectra.frequency.SED`.
        cl : callable
            :math:`C_\ell`. It returns an array with shape ``(..., ell)``.
            It can be :class:`fgspectra.cl.clSpectrum`
        """
        super().__init__(**kwargs)

        self.sed = sed
        self.cl = cl

    def eval(self, nus, bps, ell, theta):
        r"""
        Parameters
        ----------
        nus : jnp.ndarray
        bps : list[tuple[jnp.ndarray,jnp.ndarray,jnp.ndarray]]
            A list of pre-computed bandpass weights.
        ell : jnp.ndarray
        theta : jnp.ndarray
            A vector of parameters, expecting to have the SED parameters first
            and the multipole parameters second.
        """
        theta_sed = theta[: self.sed.n]
        theta_cl = theta[self.sed.n :]

        cl = self.cl(ell, *theta_cl)

        f_nus = jnp.zeros((len(ell), len(nus)))
        for i, (nu, bp) in enumerate(zip(nus, bps)):
            f_nu = self.sed(nu, *theta_sed)
            f_nus = f_nus.at[:, i].set(_bp_int(f_nu, bp))
        return jnp.einsum("...i,...ij,...il->...ijl", cl, f_nus, f_nus)

    @property
    def n(self):
        return self.cl.n + self.sed.n


class FactorizedCrossSpectrumTE(Model):
    r"""Factorized cross-spectrum for TE models.

    Cross-spectrum of a single component for which the frequency and multipole
    scaling are independent and factorizable. Distinguishes between a
    temperature and polarization SED so that they can be different, or have
    different beams/bandpasses.

    .. math:: xC_\ell^{(ij)} = A * C_\ell * f^T(\nu_i) * f^E(\nu_j)
    """

    def __init__(self, sedT, sedE, cl, **kwargs):
        super().__init__(**kwargs)

        self.sedT = sedT
        self.sedE = sedE
        self.cl = cl

    def eval(self, nus, bps, ell, theta):
        theta_sedT = theta[: self.sedT.n]
        theta_sedE = theta[self.sedT.n : self.sedT.n + self.sedE.n]
        theta_cl = theta[self.sedT.n + self.sedE.n :]

        cl = self.cl(ell, *theta_cl)

        f_nuT = jnp.zeros((len(ell), len(nus)))
        f_nuE = jnp.zeros((len(ell), len(nus)))

        for i, (nu, bp) in enumerate(zip(nus, bps)):
            f_nu = self.sedT(nu, *theta_sedT)
            f_nuT = f_nuT.at[:, i].set(_bp_int(f_nu, bp))

            f_nu = self.sedE(nu, *theta_sedT)
            f_nuE = f_nuE.at[:, i].set(_bp_int(f_nu, bp))

        return jnp.einsum("...i,...ij,...il->...ijl", cl, f_nuT, f_nuE)

    @property
    def n(self):
        return self.cl.n + self.sedT.n + self.sedE.n


class CorrelatedCrossSpectrum(Model):
    r"""Factorized cross-spectrum of correlated components.

    For two components, x and y, which are correlated:

    .. math::

        xC_\ell^{(ij)} = A_x C_\ell^x f^x(\nu_i) f^x(\nu_j) + A_y C_\ell^y f^y(\nu_i) f^y(\nu_j) - A_{xy} \sqrt{A_x A_y} C_\ell^{xy} ( f^x(\nu_i) f^y(\nu_j) + f^y(\nu_i) f^x(\nu_j) )
    """

    def __init__(self, sed1, sed2, cl1, cl2, clx, **kwargs):
        super().__init__(**kwargs)

        self.sed1 = sed1
        self.sed2 = sed2
        self.cl1 = cl1
        self.cl2 = cl2
        self.clx = clx

        self.n_s1 = jnp.arange(self.sed1.n)
        self.n_s2 = jnp.arange(self.sed2.n) + self.sed1.n
        self.n_p1 = jnp.arange(self.cl1.n) + self.sed1.n + self.sed2.n
        self.n_p2 = jnp.arange(self.cl2.n) + self.sed1.n + self.sed2.n + self.cl1.n
        self.n_px = (
            jnp.arange(self.clx.n) + self.sed1.n + self.sed2.n + self.cl1.n + self.cl2.n
        )

    def eval(self, nus, bps, ell, theta):
        theta_cl1 = theta[self.n_p1]
        theta_cl2 = theta[self.n_p2]
        theta_clx = theta[self.n_px]
        theta_s1 = theta[self.n_s1]
        theta_s2 = theta[self.n_s2]

        cl1 = self.cl1(ell, *theta_cl1)
        cl2 = self.cl2(ell, *theta_cl2)
        clx = self.clx(ell, *theta_clx)

        f_nus1 = jnp.zeros((len(ell), len(nus)))
        f_nus2 = jnp.zeros((len(ell), len(nus)))
        for i, (nu, bp) in enumerate(zip(nus, bps)):
            f_nu = self.sed1(nu, *theta_s1)
            f_nus1 = f_nus1.at[:, i].set(_bp_int(f_nu, bp))

            f_nu = self.sed2(nu, *theta_s2)
            f_nus2 = f_nus2.at[:, i].set(_bp_int(f_nu, bp))

        comp1 = jnp.einsum("...i,...ij,...il->...ijl", cl1, f_nus1, f_nus1)
        comp2 = jnp.einsum("...i,...ij,...il->...ijl", cl2, f_nus2, f_nus2)
        compx = (
            clx[:, None, None]
            * jnp.sqrt(self.cl1.amp(*theta_cl1) * self.cl2.amp(*theta_cl2))
            * (
                f_nus1[:, :, None] * f_nus2[:, None, :]
                + f_nus1[:, None, :] * f_nus2[:, :, None]
            )
        )

        return comp1 + comp2 - compx

    @property
    def n(self):
        return self.cl1.n + self.cl2.n + self.clx.n + self.sed1.n + self.sed2.n
