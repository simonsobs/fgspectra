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


T_CMB = 2.72548
H_OVER_KT_CMB = constants.h * 1e9 / constants.k / T_CMB

def _to_cmb(nu):
    x = H_OVER_KT_CMB * nu
    return (np.expm1(x) / x)**2 / np.exp(x)


def convert_to_k_cmb_unless_rj_is_true(f):
    """ Convert to CMB units by defaults
    
    Unless RJ=TRUE in the arguments, convert the output in CMB units,
    making use of the ``nu`` keyword argument. 
    If `nu_0` is among the arguments, normalize the conversion factor at that
    frequency to one.
    """
    def f_cmb_by_default(self, **kwargs):
        # We'll add RJ option in the future
        #if 'RJ' in kwargs:
            #use_rj = kwargs['RJ']
            #del kwargs['RJ']
            #if use_rj:
                #return f(self, **kwargs)
        factor = _to_cmb(kwargs['nu'])
        if 'nu_0' in kwargs:
            factor /= _to_cmb(kwargs['nu_0'])
        return f(self, **kwargs) * factor
    return f_cmb_by_default



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

    @convert_to_k_cmb_unless_rj_is_true
    def __call__(self, nu=None, beta=None, nu_0=None):
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
    @convert_to_k_cmb_unless_rj_is_true
    def __call__(self, nu=None, nu_0=None, temp=None, beta=None):
        """ Evaluation of the SED

        Parameters
        ----------
        nu: float or array
            Frequency in GHz.
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
        x = 1e+9 * constants.h * nu / (constants.k * temp)
        x_0 = 1e+9 * constants.h * nu_0 / (constants.k * temp)
        return (nu / nu_0)**(beta + 1.0) * np.expm1(x_0) / np.expm1(x)


class CIB(ModifiedBlackBody):
    """ Alias of :class:`ModifiedBlackBOdy`
    """
    pass


class ThermalSZ(SED):
    r""" Thermal Sunyaev-Zel'dovich in K_CMB

    This class implements the

    .. math:: f(\nu) = x \coth(x/2) - 4

    where :math:`x = h \nu / k_B T_CMB`
    """

    @staticmethod
    def f(nu):
        x = constants.h * (nu * 1e9) / (constants.k * T_CMB)
        return (x / np.tanh(x / 2.0) - 4.0)

    def __call__(self, nu, nu_0):
        """Compute the SED with the given frequency and parameters.

        nu : float
            Frequency in GHz.
        T_CMB (optional) : float
        """
        return ThermalSZ.f(nu, T_CMB) / ThermalSZ.f(nu_0, T_CMB)


class UnitSED(SED):
    """Frequency-independent component."""

    def __call__(self, nu, *args):
        return np.ones_like(np.array(nu))


class Join(SED):

    def __init__(self, *seds):
        self._seds = seds

    def __call__(self, kwargs_seq=None):
        """Compute the SED with the given frequency and parameters.

        *kwargss
            The length of ``kwargss`` has to be equal to the number of SEDs
            joined. ``kwargss[i]`` is a dictionary containing the keyword
            arguments of the ``i``-th SED.
        """
        seds = [sed(**kwargs) for sed, kwargs in zip(self._seds, kwargs_seq)]
        res = np.empty((len(seds),) + np.broadcast(*seds).shape)
        for i in range(len(seds)):
            res[i] = seds[i]
        return res

