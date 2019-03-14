# -*- coding: utf-8 -*-
r"""
Frequency-dependent foreground components.

This module implements the frequency-dependent component of common foreground
contaminants. This consist only of factorizable components -- i.e. the
components in this module describe :math:`f(\nu)` for which the component X has
spectrum

.. math:: C_{\ell}^{X} = f(\nu) C^X_{0,\ell}

This module draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).
"""
import numpy as np
from scipy import constants


class PowerSpectrum:
    """Base class for frequency dependent components."""

    def __init__(self, name=''):
        """Initialize the component."""
        self.spec_name = name
        return

    def __call__(self, ell, **kwargs):
        """Make component objects callable."""
        return self.powspec(ell, **kwargs)


class tSZ1(PowerSpectrum):
    """PowerSpectrum for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self):
        """Intialize object with parameters."""
        return

    def powspec(ell):
        """Compute the power spectrum with the given ell and parameters."""
        return ell


# Power law in ell

# Extragalactic FGs: from tile-c?? or Erminia / Jo

# CMB template cls?
