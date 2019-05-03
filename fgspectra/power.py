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


def _get_power_file(model):
    """ File path for the named model
    """
    data_path = pkg_resources.resource_filename('fgspectra', 'data/')
    filename = os.path.join(data_path, 'cl_%s.dat'%model)
    if os.path.exists(filename):
        return filename
    raise ValueError('No template for model '+model)


class PowerSpectrum(ABC):
    """Base class for frequency dependent components."""

    @abstractmethod
    def __call__(self, ell, *args):
        """Make component objects callable."""
        pass


class PowerSpectrumFromFile(PowerSpectrum):
    """Power spectrum loaded from file(s)

    Parameters
    ----------
    filenames: array_like of strings
        File(s) to load. It can be a string or any (nested) sequence of strings

    Examples
    --------

    >>> ell = range(5)

    Power spectrum of a single file

    >>> my_file = 'cl.dat'
    >>> ps = PowerSpectrumFromFile(my_file)
    >>> ps(ell).shape
    (5)
    >>> ps = PowerSpectrumFromFile([my_file])  # List
    >>> ps(ell).shape
    (1, 5)

    Two correlated components

    >>> my_files = [['cl_comp1.dat', 'cl_comp1xcomp2.dat'],
    ...             ['cl_comp1xcomp2.dat', 'cl_comp2.dat']]
    >>> ps = PowerSpectrumFromFile(my_files)
    >>> ps(ell).shape
    (2, 2, 5)

    """

    def __init__(self, filenames):
        """

        The file format should be two columns, ell and the spectrum.
        """
        filenames = np.array(filenames)
        self._cl = np.empty(filenames.shape+(0,))

        for i, filename in np.ndenumerate(filenames):
            ell, spec = np.genfromtxt(filename, unpack=True)
            ell = ell.astype(int)
            # Make sure that the new spectrum fits self._cl
            n_missing_ells = ell.max() + 1 - self._cl.shape[-1]
            if n_missing_ells > 0:
                self._cl = np.pad(self._cl, ((0,0), (0, n_missing_ells)),
                                  mode='constant', constant_values=0)

            self._cl[(i,)+(ell,)] = spec


    def __call__(self, ell, ell_0=3000):
        """Compute the power spectrum with the given ell and parameters."""
        return self._cl[..., ell] / self._cl[..., ell_0]


class tSZ_150_bat(PowerSpectrumFromFile):
    """PowerSpectrum for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self):
        """Intialize object with parameters."""
        super().__init__([_get_power_file('tsz_150_bat')])


class kSZ_bat(PowerSpectrumFromFile):
    """PowerSpectrum for Kinematic Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self):
        """Intialize object with parameters."""
        super().__init__([_get_power_file('ksz_bat')])


class SZxCIB(PowerSpectrumFromFile):
    """PowerSpectrum for SZxCIB (Dunkley et al. 2013)."""

    def __init__(self):
        """Intialize object with parameters."""
        files = [[_get_power_file('tsz_150_bat'), _get_power_file('sz_x_cib')],
                 [_get_power_file('sz_x_cib'), _get_power_file('cib')]]
        super().__init__(files)


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
