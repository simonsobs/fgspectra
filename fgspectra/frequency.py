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


class FrequencySpectrum:
    """Base class for frequency dependent components."""

    def __init__(self, sed_name=''):
        """Initialize the component."""
        self.sed_name = sed_name
        return

    def __call__(self, nu, **kwargs):
        """Make component objects callable."""
        return self.sed(nu, **kwargs)


class tSZ1(FrequencySpectrum):
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

    def __init__(self):
        """Intialize object with parameters."""
        self.sed_name = "tSZ1"

    def sed(self, nu, **kwargs):
        """Compute the SED with the given frequency and parameters."""
        TCMB = 2.725  # TODO: move TCMB to somewhere else
        x = constants.h * nu / (constants.k * TCMB)
        return x * np.cosh(x/2.0) / np.sinh(x/2.0) - 4.0


def calc_tSZ1(nu, *args, **kwargs):
    """Instantiate the right object and calls it, for convenience."""
    return nu

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
