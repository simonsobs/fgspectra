r"""
Models of cross-spectra

This module draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).
"""
from abc import ABC, abstractmethod
import numpy as np
from . import frequency as fgf
from . import power as fgp


class CrossSpectrum(ABC):
    """Base class for cross-spectra."""

    @abstractmethod
    def __call__(self, *args):
        """Make it callable."""
        pass


class FactorizedCrossSpectrum(CrossSpectrum):
    r"""Factorized cross-spectrum

    Cross-spectrum of **one** component for which the scaling in frequency
    and in multipoles are factorizable

    .. math:: xC_{\ell}^{(ij)} = f(\nu_j) f(\nu_i) C_{\ell}

    Parameters
    ----------
    sed : callable
        :math:`f(\nu)`. It returns an array with shape ``(..., freq)``.
        It can be :class:`fgspectra.frequency.SED`.
    cl_args : callable
        :math:`C_\ell`. It returns an array with shape ``(..., ell)``.
        It can be :class:`fgspectra.power.PowerSpectrum`

    Note
    ----
    The two (optional) sets of extra dimensions ``...`` must be
    broadcast-compatible.
    """

    def __init__(self, sed, cl):
        self._sed = sed
        self._cl = cl

    def __call__(self, sed_args, cl_args):
        """Compute the model at frequency and ell combinations.

        Parameters
        ----------
        sed_args : list
            Arguments for which the `sed` is evaluated.
        cl_args : list
            Arguments for which the `cl` is evaluated.

        Returns
        -------
        cross : ndarray
            Cross-spectrum. The shape is ``(..., freq, freq, ell)``.
        """
        f_nu = self._sed(*sed_args)[..., np.newaxis]
        return f_nu[..., np.newaxis] * f_nu * self._cl(*cl_args)


class CorrelatedFactorizedCrossSpectrum(CrossSpectrum):
    r"""Factorized cross-spectrum

    Cross-spectrum of multiple correlated components for which the scaling in
    frequency and in multipoles are factorizable.

    .. math::

        xC_\ell^{\nu_i\ \nu_j} = \sum_{kl} f^k(\nu_j) f^n(\nu_i) C^{kn}_\ell

    where :math:`k` and :math:`n` are component indices.

    Parameters
    ----------
    sed : callable
        :math:`f(\nu)`. It returns an array with shape ``(comp, ..., freq)``.
        It can be :class:`fgspectra.frequency.SED`.
    cl_args : callable
        :math:`C_\ell`. It returns an array with shape
        ``(..., comp, comp, ell)``.
        It can be :class:`fgspectra.power.PowerSpectrum`

    Note
    ----
    The two (optional) sets of extra dimensions ``...`` must be
    broadcast-compatible.
    """

    def __init__(self, sed, cl):
        self._sed = sed
        self._cl = cl

    def __call__(self, sed_args, cl_args):
        """Compute the model at frequency and ell combinations.

        Parameters
        ----------
        sed_args : list
            Arguments for which the `sed` is evaluated.
        cl_args : list
            Arguments for which the `cl` is evaluated.

        Returns
        -------
        cross : ndarray
            Cross-spectrum. The shape is ``(..., freq, freq, ell)``.
        """

        f_nu = self._sed(*sed_args)
        cl = self._cl(*cl_args)
        for _ in range(3-cl.ndim):
            cl = cl[np.newaxis, ...]
        return np.einsum('k...i,n...j,...knl->...ijl',
                         f_nu, f_nu, cl)

        # f_nu = self._sed(*sed_args)
        # return np.einsum('k...i,n...j,...knl->...ijl',
        #                  f_nu, f_nu, self._cl(*cl_args))


class PowerLaw(FactorizedCrossSpectrum):
    """ Single Power law in both fequency and multipoles

    See :class:`fgspectra.frequency.PowerLaw` for the frequency dependence and
    :class:`fgspectra.power.PowerLaw` for the ell-dependence.
    """

    def __init__(self):
        super().__init__(fgf.PowerLaw(), fgp.PowerLaw())
