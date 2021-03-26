r"""
Models of cross-spectra

This module draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).
"""
from abc import ABC, abstractmethod
import numpy as np
from . import frequency as fgf
from . import power as fgp
from .model import Model


class Sum(Model):
    """ Sum the cross-spectra of uncorrelated components
    """

    def __init__(self, *crosses, **kwargs):
        """

        Parameters
        ----------
        *sed:
            Sequence of SED models to be joined together
        """
        self._crosses = crosses
        self.set_defaults(**kwargs)

    def set_defaults(self, **kwargs):
        if 'kwseq' in kwargs:
            for cross, cross_kwargs in zip(self._crosses, kwargs['kwseq']):
                cross.set_defaults(**cross_kwargs)

    def _get_repr(self):
        return {type(self).__name__:
                    [cross._get_repr() for cross in self._crosses]}

    @property
    def defaults(self):
        return {'kwseq': [cross.defaults for cross in self._crosses]}

    def eval(self, kwseq=None):
        """Compute the sum of the cross-spectra

        *kwseq:
            The length of ``kwseq`` has to be equal to the number of
            cross-spectra summed. ``kwseq[i]`` is a dictionary containing the
            keyword arguments of the ``i``-th cross-spectrum.
        """
        if kwseq:
            crosses = (cross(**kwargs)
                       for cross, kwargs in zip(self._crosses, kwseq))
        else:  # Handles the case in which no parameter has to be passed
            crosses = (cross() for cross in self._crosses)

        res = next(crosses)
        for cross_res in crosses:
            res = res + cross_res  # Unnecessary copies can be avoided

        return res


    def diff(self, kwseq=None):
        """Compute the derivative of the sum of the cross-spectra

        *kwseq:
            The length of ``kwseq`` has to be equal to the number of
            cross-spectra summed. ``kwseq[i]`` is a dictionary containing the
            keyword arguments of the ``i``-th cross-spectrum.
        """
        if kwseq:
            crosses_diff = [cross.diff(**kwargs)
                            for cross, kwargs in zip(self._crosses, kwseq)]
        else:  # Handles the case in which no parameter has to be passed
            crosses_diff = [cross.diff() for cross in self._crosses]

        return {'kwseq':crosses_diff}


    def eval_terms(self, kwseq=None):
        """Compute the sum of the cross-spectra

        *kwseq:
            The length of ``kwseq`` has to be equal to the number of
            cross-spectra summed. ``kwseq[i]`` is a dictionary containing the
            keyword arguments of the ``i``-th cross-spectrum.
        """
        if kwseq:
            return [cross(**kwargs) for cross, kwargs in zip(self._crosses, kwseq)]
        return []


class FactorizedCrossSpectrum(Model):
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

    def __init__(self, sed, cl, **kwargs):
        self._sed = sed
        self._cl = cl
        self.set_defaults(**kwargs)

    def set_defaults(self, **kwargs):
        if 'sed_kwargs' in kwargs:
            self._sed.set_defaults(**kwargs['sed_kwargs'])
        if 'cl_kwargs' in kwargs:
            self._cl.set_defaults(**kwargs['cl_kwargs'])

    @property
    def defaults(self):
        return {
            'sed_kwargs': self._sed.defaults,
            'cl_kwargs': self._cl.defaults
        }

    def _get_repr(self):
        sed_repr = self._sed._get_repr()
        key = list(sed_repr.keys())[0]
        sed_repr[key + ' (SED)'] = sed_repr.pop(key)

        cl_repr = self._cl._get_repr()
        key = list(cl_repr.keys())[0]
        cl_repr[key + ' (Cl)'] = cl_repr.pop(key)

        return {type(self).__name__: [sed_repr, cl_repr]}

    def eval(self, sed_kwargs={}, cl_kwargs={}):
        """Compute the model at frequency and ell combinations.

        Parameters
        ----------
        sed_kwargs : list
            Arguments for which the `sed` is evaluated.
        cl_kwargs : list
            Arguments for which the `cl` is evaluated.

        Returns
        -------
        cross : ndarray
            Cross-spectrum. The shape is ``(..., freq, freq, ell)``.
        """
        f_nu = self._sed(**sed_kwargs)[..., np.newaxis]
        return f_nu[..., np.newaxis] * f_nu * self._cl(**cl_kwargs)

    def diff(self, sed_kwargs={}, cl_kwargs={}):
        """Compute the derivative of the model with respect to every
        parameters.

        Parameters
        ----------
        sed_kwargs : dict
            Arguments for which the `sed` is evaluated.
        cl_kwargs : dict
            Arguments for which the `cl` is evaluated.

        Returns
        -------
        diff : dict
            dict with same keys as the parameters passed to diff which stores derivatives with respect to parameters
        """
        sed_diff = self._sed.diff(**sed_kwargs)
        sed = self._sed(**sed_kwargs) #shape of sed is ``(..., freq)``
        cl_diff = self._cl.diff(**cl_kwargs)
        cl = self._cl(**cl_kwargs) #shape of cls is ``(..., ell)``
        tot_diff_sed = {}
        tot_diff_cl = {}
        for param in sed_diff.keys():
            if sed_diff[param] is None:
                tot_diff_sed[param] = None
            else:
                diff = sed_diff[param] #shape of diff is ``(param,...,freq)``
                tot_diff_sed[param] = np.einsum('...i,p...j,...l->p...ijl', sed, diff, cl) + \
                                      np.einsum('p...i,...j,...l->p...ijl', diff, sed, cl)

        for param in cl_diff.keys():
            if cl_diff[param] is None:
                tot_diff_cl[param] = None
            else :
                diff = cl_diff[param]  # shape of diff is ``(param,...,ell)``
                tot_diff_cl[param] = np.einsum('...i,...j,p...l->p...ijl',sed, sed, diff)

        return {'sed_kwargs':tot_diff_sed, 'cl_kwargs':tot_diff_cl}


class CorrelatedFactorizedCrossSpectrum(FactorizedCrossSpectrum):
    r"""Factorized cross-spectrum of correlated components

    Cross-spectrum of multiple correlated components for which the scaling in
    frequency and in multipoles are factorizable.

    .. math::

        xC_\ell^{\nu_i\ \nu_j} = \sum_{kn} f^k(\nu_j) f^n(\nu_i) C^{kn}_\ell

    where :math:`k` and :math:`n` are component indices.

    Parameters
    ----------
    sed : callable
        :math:`f(\nu)`. It returns an array with shape ``(comp, ..., freq)``.
        It can be :class:`fgspectra.frequency.SED`.
    cl_args : callable
        :math:`C_\ell`. It returns an array with shape
        ``(..., comp, comp, ell)``.
        It can, for example, any of the models in :class:`fgspectra.power`.

    Note
    ----
    The two (optional) sets of extra dimensions ``...`` must be
    broadcast-compatible.
    """

    def __init__(self, sed, cl, **kwargs):
        self._sed = sed
        self._cl = cl
        self.set_defaults(**kwargs)

    def eval(self, sed_kwargs={}, cl_kwargs={}):
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

        f_nu = self._sed(**sed_kwargs)
        return np.einsum('k...i,n...j,...knl->...ijl',
                         f_nu, f_nu, self._cl(**cl_kwargs))


class PowerLaw(FactorizedCrossSpectrum):
    """ Single Power law in both fequency and multipoles

    See :class:`fgspectra.frequency.PowerLaw` for the frequency dependence and
    :class:`fgspectra.power.PowerLaw` for the ell-dependence.
    """

    def __init__(self, **kwargs):
        super().__init__(fgf.PowerLaw(), fgp.PowerLaw())
        self.set_defaults(**kwargs)


class CorrelatedPowerLaw(CorrelatedFactorizedCrossSpectrum):
    """ Correlated Power law in both fequency and multipoles

    See :class:`fgspectra.frequency.PowerLaw` for the frequency dependence and
    :class:`fgspectra.power.PowerLaw` for the ell-dependence.
    """

    def __init__(self, **kwargs):
        super().__init__(fgf.PowerLaw(), fgp.CorrelatedPowerLaws())
        self.set_defaults(**kwargs)


class CorrelatedDustSynchrotron(CorrelatedFactorizedCrossSpectrum):
    """ CorrelatedDustSynchrotron
    
    Correlated power law and modified black body, both with power law amplitude
    """

    def __init__(self, **kwargs):
        super().__init__(
            fgf.Join(fgf.ModifiedBlackBody(), fgf.PowerLaw()),
            fgp.CorrelatedPowerLaws()
        )
        self.set_defaults(**kwargs)


class SZxCIB(CorrelatedFactorizedCrossSpectrum):
    
    def __init__(self, **kwargs):
        sed = fgf.Join(fgf.ThermalSZ(), fgf.CIB())
        super().__init__(sed, fgp.SZxCIB_Addison2012())
        self.set_defaults(**kwargs)

class WhiteNoise(Model):
    """White noise"""

    def eval(self, nu=None, ell=None, nwhite=None):
        """ Evaluation of the model

        Parameters
        ----------
        nu : float or array
            Frequency at which model will be evaluated. If array, the shape
            is ``(freq)``.
        ell : float or array
            Multipoles at which model will be evaluated. If array, the shape
            is ``(ells)``.
        nwhite : ndarray
            white noise levels, shape is ``(freqs)``
        Returns
        -------
        cov : ndarray
            Shape is ``(ells,freqs,freqs)``
        """
        if type(nu) in (float, int):
            nu = [nu]
        n_freqs = len(nu)
        if type(ell) in (float, int):
            ell = [ell]
        n_ell = len(ell)
        if type(nwhite) in (float, int):
            nwhite = [nwhite]
        if len(nwhite) == 1 and n_freqs > 1:
            print('Expected {:d} noise levels but got 1. Will use the same '
                  'at all frequencies'.format(n_freqs))
            nwhite = np.ones(n_freqs) * nwhite
        elif len(nwhite) != n_freqs:
            print('Got {:d} white noise levels, expected {:d}'.format(
                len(nwhite), n_freqs))
        res = np.broadcast_to(np.diag(nwhite**2), (n_ell, n_freqs, n_freqs))
        return np.transpose(res, (1, 2, 0))

    def diff(self, nu=None, ell=None, nwhite=None):
        """ Evaluation of the derivative of the model

        Parameters
        ----------
        nu : float or array
            Frequency at which model will be evaluated. If array, the shape
            is ``(freq)``.
        ell : float or array
            Multipoles at which model will be evaluated. If array, the shape
            is ``(ells)``.
        nwhite : ndarray
            white noise levels, shape is ``(freqs)``
        Returns
        -------
        diff: dict
            Each key corresponds to the the derivative with respect to a parameter.
        """
        (nu, ell, nwhite) = self._replace_none_args((nu, ell, nwhite))
        if type(nu) in (float, int):
            nu = [nu]
        n_freqs = len(nu)
        if type(ell) in (float, int):
            ell = [ell]
        n_ell = len(ell)
        if type(nwhite) in (float, int):
            nwhite = [nwhite]
        if len(nwhite) == 1 and n_freqs > 1:
            print('Expected {:d} noise levels but got 1. Will use the same '
                  'at all frequencies'.format(n_freqs))
            nwhite = np.ones(n_freqs) * nwhite
        elif len(nwhite) != n_freqs:
            print('Got {:d} white noise levels, expected {:d}'.format(
                len(nwhite), n_freqs))
        diff_nwhite_ell = np.zeros((n_freqs, n_freqs, n_freqs))
        np.fill_diagonal(diff_nwhite_ell,2.*nwhite)
        diff_nwhite = np.broadcast_to(diff_nwhite_ell,
                                      (n_ell, n_freqs, n_freqs, n_freqs)).T
        return {'nu':None, 'ell':None, 'nwhite':diff_nwhite}
