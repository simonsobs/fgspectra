r"""
Models of cross-spectra

This module draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).
"""
import numpy as np
from . import frequency as fgf
from . import power as fgp
from .model import Model


class Sum(Model):
    """Sum the cross-spectra of uncorrelated components"""

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
        if "kwseq" in kwargs:
            for cross, cross_kwargs in zip(self._crosses, kwargs["kwseq"]):
                cross.set_defaults(**cross_kwargs)

    def _get_repr(self):
        return {type(self).__name__: [cross._get_repr() for cross in self._crosses]}

    @property
    def defaults(self):
        return {"kwseq": [cross.defaults for cross in self._crosses]}

    def eval(self, kwseq=None):
        """Compute the sum of the cross-spectra

        *kwseq:
            The length of ``kwseq`` has to be equal to the number of
            cross-spectra summed. ``kwseq[i]`` is a dictionary containing the
            keyword arguments of the ``i``-th cross-spectrum.
        """
        if kwseq:
            crosses = (cross(**kwargs) for cross, kwargs in zip(self._crosses, kwseq))
        else:  # Handles the case in which no parameter has to be passed
            crosses = (cross() for cross in self._crosses)

        res = next(crosses)
        for cross_res in crosses:
            res = res + cross_res  # Unnecessary copies can be avoided

        return res

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
        if "sed_kwargs" in kwargs:
            self._sed.set_defaults(**kwargs["sed_kwargs"])
        if "cl_kwargs" in kwargs:
            self._cl.set_defaults(**kwargs["cl_kwargs"])

    @property
    def defaults(self):
        return {"sed_kwargs": self._sed.defaults, "cl_kwargs": self._cl.defaults}

    def _get_repr(self):
        sed_repr = self._sed._get_repr()
        key = list(sed_repr.keys())[0]
        sed_repr[key + " (SED)"] = sed_repr.pop(key)

        cl_repr = self._cl._get_repr()
        key = list(cl_repr.keys())[0]
        cl_repr[key + " (Cl)"] = cl_repr.pop(key)

        return {type(self).__name__: [sed_repr, cl_repr]}

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
        cl = self._cl(**cl_kwargs)
        if f_nu.shape[0] != cl.shape[-1] or (f_nu.shape[0] == 1 and cl.shape[-1] == 1):
            f_nu = f_nu[np.newaxis]

        return np.einsum("l...i,l...j,...l->...ijl", f_nu, f_nu, cl)


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
        cl = self._cl(**cl_kwargs)
        if f_nu.shape[0] != cl.shape[-1]:
            f_nu = f_nu[:, np.newaxis]

        return np.einsum("kl...i,nl...j,...knl->...ijl", f_nu, f_nu, cl)


class PowerLaw(FactorizedCrossSpectrum):
    """Single Power law in both fequency and multipoles

    See :class:`fgspectra.frequency.PowerLaw` for the frequency dependence and
    :class:`fgspectra.power.PowerLaw` for the ell-dependence.
    """

    def __init__(self, **kwargs):
        super().__init__(fgf.PowerLaw(), fgp.PowerLaw())
        self.set_defaults(**kwargs)


class CorrelatedPowerLaw(CorrelatedFactorizedCrossSpectrum):
    """Correlated Power law in both fequency and multipoles

    See :class:`fgspectra.frequency.PowerLaw` for the frequency dependence and
    :class:`fgspectra.power.PowerLaw` for the ell-dependence.
    """

    def __init__(self, **kwargs):
        super().__init__(fgf.PowerLaw(), fgp.CorrelatedPowerLaws())
        self.set_defaults(**kwargs)


class CorrelatedDustSynchrotron(CorrelatedFactorizedCrossSpectrum):
    """CorrelatedDustSynchrotron

    Correlated power law and modified black body, both with power law amplitude
    """

    def __init__(self, **kwargs):
        super().__init__(
            fgf.Join(fgf.ModifiedBlackBody(), fgf.PowerLaw()), fgp.CorrelatedPowerLaws()
        )
        self.set_defaults(**kwargs)


class SZxCIB(CorrelatedFactorizedCrossSpectrum):
    def __init__(self, **kwargs):
        sed = fgf.Join(fgf.ThermalSZ(), fgf.CIB())
        super().__init__(sed, fgp.SZxCIB_Addison2012())
        self.set_defaults(**kwargs)


class SZxCIB_Choi2020(CorrelatedFactorizedCrossSpectrum):
    def __init__(self, **kwargs):
        sed = fgf.Join(fgf.ThermalSZ(), fgf.CIB())
        power_spectra = [
            fgp.PowerSpectrumFromFile(fgp._get_power_file("tsz_150_bat")),
            fgp.PowerSpectrumFromFile(fgp._get_power_file("cib_Choi2020")),
            fgp.PowerSpectrumFromFile(fgp._get_power_file("sz_x_cib")),
        ]
        cl = fgp.PowerSpectraAndCovariance(*power_spectra)
        super().__init__(sed, cl)
        self.set_defaults(**kwargs)
