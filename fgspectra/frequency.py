# -*- coding: utf-8 -*-
r"""
Frequency-dependent foreground components.

This module implements the frequency-dependent component of common foreground
contaminants.

This package draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).
"""

import numpy as np
from scipy import constants
from .model import Model
from functools import wraps


T_CMB = 2.72548
H_OVER_KT_CMB = constants.h * 1e9 / constants.k / T_CMB

def _rj2cmb(nu):
    x = H_OVER_KT_CMB * nu
    return (np.expm1(x) / x)**2 / np.exp(x)


class PowerLaw(Model):
    r""" Power Law

    .. math:: f(\nu) = (\nu / \nu_0)^{\beta}
    """
    def eval(self, nu=None, beta=None, nu_0=None):
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
        return (nu / nu_0)**beta * (_rj2cmb(nu) / _rj2cmb(nu_0))


class Synchrotron(PowerLaw):
    """ Alias of :class:`PowerLaw`
    """
    pass


class RadioSourcesFluxCut(Model):
    """ Radio point source power law with amplitude from flux cut
    
    .. math:: f(\nu) = amp (\nu / \nu_0)^{\beta}
    .. math:: amp = 
    """
    def eval(self, nu=None, nu_0=None, fluxcut=None, beta=None):
        """ Evaluation of the SED

        Parameters
        ----------
        nu: float or array
            Frequency in GHz.
        fluxcut: float or array
            Flux cut in Jy
        beta: float or array
            Spectral index. Normally -2.5
        nu_0: float
            Reference frequency in Hz.

        Returns
        -------
        sed: ndarray
            The last dimension is the frequency dependence.
            The leading dimensions are the broadcast between the hypothetic
            dimensions of `beta` and `fluxcut`.
        """
        fluxcut = np.array(temp)[..., np.newaxis]
        beta = np.array(beta)[..., np.newaxis]
    
        # Fill here with flux cut equations    
    
        return
        

class ModifiedBlackBody(Model):
    r""" Modified black body in K_RJ

    .. math:: f(\nu) = (\nu / \nu_0)^{\beta + 1} / (e^x - 1)

    where :math:`x = h \nu / k_B T_d`
    """
    def eval(self, nu=None, nu_0=None, temp=None, beta=None):
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
        res = (nu / nu_0)**(beta + 1.0) * np.expm1(x_0) / np.expm1(x)
        return res * (_rj2cmb(nu) / _rj2cmb(nu_0))


class CIB(ModifiedBlackBody):
    """ Alias of :class:`ModifiedBlackBOdy`
    """
    pass


class ThermalSZ(Model):
    r""" Thermal Sunyaev-Zel'dovich in K_CMB

    This class implements the

    .. math:: f(\nu) = x \coth(x/2) - 4

    where :math:`x = h \nu / k_B T_CMB`
    """

    @staticmethod
    def f(nu):
        x = constants.h * (nu * 1e9) / (constants.k * T_CMB)
        return (x / np.tanh(x / 2.0) - 4.0)

    def eval(self, nu=None, nu_0=None):
        """Compute the SED with the given frequency and parameters.

        nu : float
            Frequency in GHz.
        T_CMB (optional) : float
        """
        return ThermalSZ.f(nu) / ThermalSZ.f(nu_0)


class FreeFree(Model):
    r""" Free-free

    .. math:: f(\nu) = EM * ( 1 + log( 1 + (\nu_{ff} / \nu)^{3/\pi} ) )
    .. math:: \nu_{ff} = 255.33e9 * (Te / 1000)^{3/2}
    """
    def eval(self, nu=None, EM=None, Te=None):
        """ Evaluation of the SED

        Parameters
        ----------
        nu: float or array
            Frequency in the same units as `nu_0`. If array, the shape is
            ``(freq)``.
        EM: float or array
            Emission measure in cm^-6 pc (usually around 300). If array, the shape is ``(...)``.
        Te: float or array
            Electron temperature (typically around 7000). If array, the shape is ``(...)``.

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

        - Free-free emission in temperature.

        """
        EM  = np.array(EM)[..., np.newaxis]
        Te  = np.array(Te)[..., np.newaxis]
        Teff = (Te / 1.e3)**(1.5)
        nuff = 255.33e9 * Teff
        gff = 1. + np.log(1. + (nuff / nu)**(np.sqrt(3) / np.pi))
        print("warning: I need to check the units on this")
        return EM * gff

class ConstantSED(Model):
    """Frequency-independent component."""

    def eval(self, nu=None, amp=1.):
        """ Evaluation of the SED

        Parameters
        ----------
        nu: float or array
            It just determines the shape of the output.
        amp: float or array
            Amplitude (or set of amplitudes) of the constant SED.

        Returns
        -------
        sed: ndarray
            If `nu` is an array, the shape is ``amp.shape + (freq)``.
            If `nu` is scalar, the shape is ``amp.shape + (1)``.
            Note that the last dimension is guaranteed to be the frequency.
        """
        amp = np.array(amp)[..., np.newaxis]
        return amp * np.ones_like(np.array(nu))


class Join(Model):
    """ Join several SED models together
    """

    def __init__(self, *seds, **kwargs):
        """ Join several SED models together

        Parameters
        ----------
        *sed:
            Sequence of SED models to be joined together
        """
        self._seds = seds
        self.set_defaults(**kwargs)

    def set_defaults(self, **kwargs):
        if 'kwseq' in kwargs:
            for sed, sed_kwargs in zip(self._seds, kwargs['kwseq']):
                sed.set_defaults(**sed_kwargs)

    def _get_repr(self):
        return {type(self).__name__: [sed._get_repr() for sed in self._seds]}

    @property
    def defaults(self):
        return {'kwseq': [sed.defaults for sed in self._seds]}

    def eval(self, kwseq=None):
        """Compute the SED with the given frequency and parameters.

        *kwseq
            The length of ``kwseq`` has to be equal to the number of SEDs
            joined. ``kwseq[i]`` is a dictionary containing the keyword
            arguments of the ``i``-th SED.
        """
        if kwseq:
            seds = [sed(**kwargs) for sed, kwargs in zip(self._seds, kwseq)]
        else:  # Handles the case in which no parameter has to be passed
            seds = [sed() for sed in self._seds]
        res = np.empty((len(seds),) + np.broadcast(*seds).shape)
        for i in range(len(seds)):
            res[i] = seds[i]
        return res
