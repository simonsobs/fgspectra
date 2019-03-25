r"""
Power spectrum

This module implements the ell-dependent component of common foreground
contaminants.

This module draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).
"""

import os
import pkg_resources
from abc import ABC, abstractmethod
import numpy as np


class PowerSpectrum(ABC):
    """Base class for frequency dependent components."""

    @abstractmethod
    def __call__(self, ell, *args):
        """Make component objects callable."""
        pass


class PowerSpectrumFromFile(PowerSpectrum):
    """Generic PowerSpectrum loaded from file."""

    def __init__(self, filename):
        """

        The file format should be two columns, ell and the spectrum.
        """
        data_path = pkg_resources.resource_filename('fgspectra', 'data/')
        file_path = os.path.join(data_path, filename)
        ell, spec = np.genfromtxt(file_path, unpack=True)
        self._cl = np.zeros(ell.max() + 1)
        self._cl[ell] = spec

    def __call__(self, ell):
        """Compute the power spectrum with the given ell and parameters."""
        return self._cl[ell]


class tSZ_150_bat(PowerSpectrumFromFile):
    """PowerSpectrum for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self):
        """Intialize object with parameters."""
        super().__init__("cl_tsz_150_bat.dat")


class kSZ_bat(PowerSpectrumFromFile):
    """PowerSpectrum for Kinematic Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self):
        """Intialize object with parameters."""
        super().__init__("cl_ksz_bat.dat")


class sz_x_cib_template(PowerSpectrumFromFile):
    """PowerSpectrum for SZxCIB (Dunkley et al. 2013)."""

    def __init__(self):
        """Intialize object with parameters."""
        super().__init__("sz_x_cib_template.dat")


class PowerLaw(PowerSpectrum):
    r""" Power law

    .. math:: C_\ell = (\ell / \ell_0)^\alpha
    """
    def __call__(self, ell, alpha, ell_0):
        """

        Parameters
        ----------
        ell: float or array
            Multipole
        alpha: float or array
            Spectral index.
        ell_0: float
            Reference ell

        Returns
        -------
        cl: ndarray
            The last dimension is ell.
            The leading dimensions are the hypothetic dimensions of `alpha`
        """
        alpha = np.array(alpha)[..., np.newaxis]
        return (ell / ell_0)**alpha


# Power law in ell

# Extragalactic FGs: from tile-c?? or Erminia / Jo

# CMB template cls?
