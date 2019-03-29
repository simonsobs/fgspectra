# -*- coding: utf-8 -*-
r"""
Frequency-dependent foreground components.

This module implements the frequency-dependent component of common foreground
contaminants. 

This package draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy import constants

# TODO: we need to figure out the unit situation.

T_CMB = 2.725


class SED(ABC):
    """Base class for frequency dependent components."""

    @abstractmethod
    def __call__(self, nu, *args):
        # Return the evaluation of the SED
        pass


class PowerLaw(SED):
    r""" Power Law

    .. math:: f(\nu) = (\nu / nu_0)^{\beta}
    """

    def __call__(self, nu, beta, nu0):
        """ Evaluation of the SED

        Parameters
        ----------
        nu: float or array
            Frequency in the same units as `nu0`. If array, the shape is
            ``(freq)``.
        beta: float or array
            Spectral index. If array, the shape is ``(...)``.
        nu0: float or array
            Reference frequency in the same units as `nu`. If array, the shape
            is ``(...)``.

        Returns
        -------
        sed: ndarray
            If `nu` is an array, the shape is ``(..., freq)``.
            If `nu` is scalar, the shape is ``(..., 1)``.
            Note that the last dimension is guaranteed to be the frequency.

        Note
        ----
        The extra dimensions ``...`` in the output are the broadcast of the
        ``...`` in the input (which are required to be broadcast-compatible).

        Examples
        --------

        - T, E and B synchrotron SEDs with the same reference frequency but
          different spectral indices. `beta` is an array with shape ``(3)``,
          `nu0` is a scalar.

        - SEDs of synchrotron and dust (approximated as power law). Both `beta`
          and `nu0` are arrays with shape ``(2)``

        """
        beta = np.array(beta)[..., np.newaxis]
        nu0 = np.array(nu0)[..., np.newaxis]
        return (nu / nu0)**beta


class Synchrotron(PowerLaw):
    """ Alias of :class:`PowerLaw`
    """
    pass


class ModifiedBlackBody(SED):
    """ Modified black body in K_RJ

    .. math:: f(\nu) = (nu / nu_0)^{\beta + 1} / (e^x - 1)

    where :math:`x = h \nu / k_B T_d`
    """
    def __call__(nu, nu_0, temp, beta):
        """ Evaluation of the SED

        Parameters
        ----------
        nu: float or array
            Frequency in Hz.
        beta: float or array
            Spectral index.
        temp: float or array
            Dust temperature.
        nu0: float
            Reference frequency in Hz.

        Returns
        -------
        sed: ndarray
            The last dimension is the frequency dependence.
            The leading dimensions are the broadcast between the hypothetic
            dimensions of `beta` and `temp`.
        """
        beta = np.array(beta)[..., np.newaxis]
        temp = np.array(temp)[..., np.newaxis]
        x = constants.h * nu / (constants.k * temp)
        return (nu / nu_0)**(beta + 1.0) / np.expm1(x)


class ThermalSZ(SED):
    r""" Thermal Sunyaev-Zel'dovich in K_CMB

    This class implements the

    .. math:: f(\nu) = x \coth(x/2) - 4

    where :math:`x = h \nu / k_B T_CMB`
    """
    def __call__(self, nu):
        """Compute the SED with the given frequency and parameters.

        nu : float
            Frequency in GHz.
        T_CMB (optional) : float
        """
        x = constants.h * (nu*1e9) / (constants.k * T_CMB)
        return x / np.tanh(x / 2.0) - 4.0


r'''
class CIB(SED):
    r"""
    SED for :math:`g(\nu)` for CIB models, where

    .. math:: g(\mu) = (\partial B_{\nu}/\partial T)^{-1} |_{T_CMB}

    Parameters
    ----------
    nu: float or array
        Frequency in Hz.
    beta: float
        Spectral index.

    Methods
    -------
    __call__(self, nu, beta)
        return the frequency scaling given by a power law.
    """
    def g(self, nu, T_CMB):
        """Convert from flux to thermodynamic units."""
        x = constants.h * (nu*1e9) / (constants.k * T_CMB)
        return constants.c**2 * constants.k * T_CMB**2 * (np.cosh(x) - 1) / (
            constants.h**2 * (nu*1e9)**4)

    def planckB(self, nu, T_d):
        """Planck function at dust temperature."""
        x = constants.h * (nu*1e9) / (constants.k * T_d)
        return 2 * constants.h * (nu*1e9)**3 / constants.c**2 / (np.exp(x) - 1)

    def sed(self, nu, beta, T_d, T_CMB):
        """Modified blackbody mu(nu, beta)"""
        return nu**beta * self.planckB(nu, T_d) * self.g(nu, T_CMB)
'''


# CMB
# blackbody, derivative BB

# Galactic foregrounds
# power law (synch), modified blackbody (dust)
# extensions of these?
# before and fgbuster

# Extragalactic
# mystery
# tile-c

# thermodynamic CMB units - defualt
# other units: Jy/sr or Rayleigh-Jeans, full antenna temp conversion?
