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

    def __call__(self, nu, beta, nu_0):
        """ Evaluation of the SED

        Parameters
        ----------
        nu: float or array
            Frequency in the same units as `nu_0`. If array, the shape is
            ``(freq)``.
        beta: float or array
            Spectral index. If array, the shape is ``(...)``.
        nu_0: float or array
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
          `nu_0` is a scalar.

        - SEDs of synchrotron and dust (approximated as power law). Both `beta`
          and `nu_0` are arrays with shape ``(2)``

        """
        beta = np.array(beta)[..., np.newaxis]
        nu_0 = np.array(nu_0)[..., np.newaxis]
        return (nu / nu_0)**beta


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
        nu_0: float
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


def g(nu, T_CMB=2.725):
    """Convert from flux to thermodynamic units."""
    x = constants.h * (nu*1e9) / (constants.k * T_CMB)
    return constants.c**2 * constants.k * T_CMB**2 * (np.cosh(x) - 1) / (
        constants.h**2 * (nu*1e9)**4)


class ThermalSZ(SED):
    r""" Thermal Sunyaev-Zel'dovich in K_CMB

    This class implements the

    .. math:: f(\nu) = x \coth(x/2) - 4

    where :math:`x = h \nu / k_B T_CMB`
    """

    @staticmethod
    def f(nu, T_CMB=2.725):
        x = constants.h * (nu*1e9) / (constants.k * T_CMB)
        return (x / np.tanh(x / 2.0) - 4.0)

    def __call__(self, nu, nu_0, T_CMB):
        """Compute the SED with the given frequency and parameters.

        nu : float
            Frequency in GHz.
        T_CMB (optional) : float
        """
        return ThermalSZ.f(nu, T_CMB) / ThermalSZ.f(nu_0, T_CMB)


class UnitSED(SED):
    """Frequency-independent component."""

    def __call__(self, nu, *args):
        # Return the evaluation of the SED
        return np.ones(np.array(nu).shape)

class CIB(SED):


    @staticmethod
    def planckB(nu, T_d):
        """Planck function at dust temperature."""
        x = constants.h * (nu*1e9) / (constants.k * T_d)
        return 2 * constants.h * (nu*1e9)**3 / constants.c**2 / (np.exp(x) - 1)

    @staticmethod
    def mu(nu, beta, T_d, T_CMB):
        return nu**beta * CIB.planckB(nu, T_d) * g(nu, T_CMB)

    def __call__(self, nu, beta, T_d, nu_0, T_CMB=2.725):
        """Compute the SED with the given frequency and parameters.

        nu : float
            Frequency in GHz.
        beta : power law parameter
        T_d : dust temperature
        T_CMB (optional) : float
        """
        return CIB.mu(nu, beta, T_d, T_CMB) / CIB.mu(nu_0, beta, T_d, T_CMB)


class tSZxCIB(SED):
    def __init__(self, effective_frequencies=False):
        """This is the only object for which it matters to have legacy
        support for effective frequencies."""
        self.effective_frequencies = effective_frequencies

    def __call__(self, nu_1, nu_2, beta, nu_0, T_d=9.7, T_CMB=2.725):
        """Compute the SED with the given frequency and parameters.

        nu : float
            Frequency in GHz.
        beta : power law parameter
        T_d : dust temperature
        T_CMB (optional) : float
        """
        beta = np.array(beta)[..., np.newaxis]
        T_d = np.array(T_d)[..., np.newaxis]
        if self.effective_frequencies:
            return np.sqrt( ThermalSZ.f(nu_1, T_CMB)[...,np.newaxis] * CIB.mu(nu_2, beta, T_d, T_CMB)
                         / (ThermalSZ.f(nu_0, T_CMB) * CIB.mu(nu_0, beta, T_d, T_CMB)))
        else:
            # NOTE: Dunkley et al. 2013 does not use both terms in equation 12
            return ( (ThermalSZ.f(nu_1, T_CMB)[...,np.newaxis] * CIB.mu(nu_2, beta, T_d, T_CMB) +
                         CIB.mu(nu_1, beta, T_d, T_CMB)[...,np.newaxis] * ThermalSZ.f(nu_2, T_CMB))
                         / (2 * ThermalSZ.f(nu_0, T_CMB) * CIB.mu(nu_0, beta, T_d, T_CMB)))


class PowerLaw_g(SED):

    def __call__(self, nu, beta, nu_0, T_CMB=2.725):
        """Compute the SED with the given frequency and parameters.

        nu : float
            Frequency in GHz.
        beta : power law parameter
        T_d : dust temperature
        T_CMB (optional) : float
        """

        beta = np.array(beta)[..., np.newaxis]
        norm = nu_0**beta * g(nu_0, T_CMB)
        return nu**beta * g(nu, T_CMB) / norm
