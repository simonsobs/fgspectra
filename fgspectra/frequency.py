# -*- coding: utf-8 -*-
r"""
Frequency-dependent foreground components.

This module implements the frequency-dependent component of common foreground
contaminants. This consist only of factorizable components -- i.e. the components
in this module describe :math:`f(\nu)` for which the component X has spectrum

.. math:: C_{\ell}^{X} = f(\nu) C^X_{0,\ell}

This package draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).
"""
import numpy as np
from scipy import constants

# TODO: we need to figure out the unit situation.

class FrequencySpectrum:
    """Base class for frequency dependent components."""

    def __init__(self, sed_name=''):
        """Initialize the component."""
        pass

    def __call__(self, nu, **kwargs):
        """Calls the object's `sed(nu, **kwargs)` method."""
        return self.sed(nu, **kwargs)


class PowerLaw(FrequencySpectrum):
    r"""
    FrequencySpectrum for a power law.
    .. math:: f(\nu) = (nu / nu_0)^{\beta}

    Parameters
    ----------
    nu: float or array
        Frequency in Hz.
    beta: float
        Spectral index.
    nu0: float
        Reference frequency in Hz.

    Methods
    -------
    __call__(self, nu, beta)
        return the frequency scaling given by a power law.
    """
    def sed(nu, beta, nu_0):
        """Compute the SED with the given frequency and parameters."""
        par = self.get_missing(kwargs)
        return (nu / nu_0)**beta


class Synchrotron(PowerLaw):
    r""" Alias of :class:`PowerLaw`
    """
    pass


class ModifiedBlackBody(FrequencySpectrum):
    r"""
    FrequencySpectrum for a modified black body.
    .. math:: f(\nu) = (nu / nu_0)^{\beta + 1} / (e^X - 1)

    where :math:`X = h \nu / k_B T_d`

    Parameters
    ----------
    nu: float or array
        Frequency in Hz.
    beta: float
        Spectral index.
    T_d: float
        Dust temperature.
    nu0: float
        Reference frequency in Hz.

    Methods
    -------
    __call__(self, nu, beta, T_d)
        return the frequency dependent component of Synchrotron.
    """
    def sed(nu, nu_0, T_d, beta):
        """Compute the SED with the given frequency and parameters."""
        x = constants.h * (nu * 1e9) / (constants.k * T_d)
        return (nu / nu_0)**(beta+1) / (np.exp(X) - 1)


class ThermalSZFreq(FrequencySpectrum):
    r"""
    FrequencySpectrum for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013).

    This class implements the

    .. math:: f(\nu) = x \coth(x/2) - 4

    where :math:`x = h \nu / k_B T_CMB`

    Methods
    -------
    __call__(self, nu)
        return the frequency dependent component of the tSZ.
    """
    def sed(self, nu, T_CMB=2.725):
        """Compute the SED with the given frequency and parameters.

        nu : float
            Frequency in GHz.
        T_CMB (optional) : float
        """
        x = constants.h * (nu*1e9) / (constants.k * T_CMB)
        return x * np.cosh(x/2.0) / np.sinh(x/2.0) - 4.0


class CIBFreq(FrequencySpectrum):
    r"""
    FrequencySpectrum for :math:`g(\nu` for CIB models, where

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
