# -*- coding: utf-8 -*-
r"""
Frequency-dependent foreground components.

This module implements the ell-dependent component of common foreground
contaminants. This consist only of factorizable components -- i.e. the
components in this module describe :math:`f(\nu)` for which the component X has
spectrum

.. math:: C_{\ell}^{X} = f(\nu) C^X_{0,\ell}

This module draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).
"""
import numpy as np
from scipy import constants
import pkg_resources
import os


class PowerSpectrum:
    """Base class for frequency dependent components."""

    def __init__(self):
        """Initialize the component."""
        pass

    def __call__(self, ell, **kwargs):
        """Make component objects callable."""
        return self.powspec(ell, **kwargs)


class PowerSpectrumFromFile(PowerSpectrum):
    """Generic PowerSpectrum loaded from file."""

    def __init__(self, filename):
        """Intialize object with parameters.

        The file format should be two columns, ell and the spectrum.
        """
        data_path = pkg_resources.resource_filename('fgspectra', 'data/')
        file_path = os.path.join(data_path, filename)
        self.data_ell, self.data_spec = np.genfromtxt(file_path, unpack=True)
        return

    def powspec(self, ell):
        """Compute the power spectrum with the given ell and parameters."""
        return np.interp(ell, self.data_ell, self.data_spec)


class tSZ_150_bat(PowerSpectrumFromFile):
    """PowerSpectrum for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self):
        """Intialize object with parameters."""
        super().__init__("cl_tsz_150_bat.dat")
        return


class kSZ_bat(PowerSpectrumFromFile):
    """PowerSpectrum for Kinematic Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self):
        """Intialize object with parameters."""
        super().__init__("cl_ksz_bat.dat")
        return


class sz_x_cib_template(PowerSpectrumFromFile):
    """PowerSpectrum for SZxCIB (Dunkley et al. 2013)."""

    def __init__(self):
        """Intialize object with parameters."""
        super().__init__("sz_x_cib_template.dat")
        return


class PowerLaw(PowerSpectrum):
    r"""
    PowerSpectrum for a power law.
    .. math:: f(\nu) = (nu / nu_0)^{\beta}

    Parameters
    ----------
    ell: float or array
        Frequency in Hz.
    beta: float
        Spectral index.
    ell_0: float
        Reference ell

    Methods
    -------
    __call__(self, ell, beta, ell_0)
        return the ell scaling given by a power law.
    """

    def powspec(self, ell, beta, ell_0):
        return (ell/ell_0)**beta




# Power law in ell

# Extragalactic FGs: from tile-c?? or Erminia / Jo

# CMB template cls?
