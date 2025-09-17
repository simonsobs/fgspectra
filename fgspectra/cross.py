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
    cl : callable
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
        if len(f_nu.shape) == len(cl.shape) == 1:
            return np.outer(f_nu, f_nu)[:, :, np.newaxis] * cl
        # f_nu.shape can be either [freq] or [ell, freq]
        if f_nu.shape[0] != cl.shape[-1] or (f_nu.shape[0] == 1 and cl.shape[-1] == 1):
            f_nu = f_nu[np.newaxis]

        return np.einsum("l...i,l...j,...l->...ijl", f_nu, f_nu, cl)


class FactorizedCrossSpectrumTE(Model):
    r"""Factorized cross-spectrum

    TE Cross-spectrum of **one** component for which the scaling in frequency
    and in multipoles are factorizable. It is useful to distinguish the T and E
    SEDs since they could be computed by integrating them with different
    beams and/or transmissions.

    .. math:: xC^TE_{\ell}^{(ij)} = f^T(\nu_j) f^E(\nu_i) C_{\ell}

    Parameters
    ----------
    sedT : callable
        The temperature SED :math:`f^T(\nu)`.
        It returns an array with shape ``(..., freq)``.
        It can be :class:`fgspectra.frequency.SED`.
    sedE : callable
        The E mode SED :math:`f^E(\nu)`.
        It returns an array with shape ``(..., freq)``.
        It can be :class:`fgspectra.frequency.SED`.
    cl_args : callable
        :math:`C_\ell`. It returns an array with shape ``(..., ell)``.
        It can be :class:`fgspectra.power.PowerSpectrum`

    Note
    ----
    The two (optional) sets of extra dimensions ``...`` must be
    broadcast-compatible.
    """

    def __init__(self, sedT, sedE, cl, **kwargs):
        self._sedT = sedT
        self._sedE = sedE
        self._cl = cl
        self.set_defaults(**kwargs)

    def set_defaults(self, **kwargs):
        if "sedT_kwargs" in kwargs:
            self._sedT.set_defaults(**kwargs["sedT_kwargs"])
        if "sedE_kwargs" in kwargs:
            self._sedE.set_defaults(**kwargs["sedE_kwargs"])
        if "cl_kwargs" in kwargs:
            self._cl.set_defaults(**kwargs["cl_kwargs"])

    @property
    def defaults(self):
        return {
            "sedT_kwargs": self._sedT.defaults,
            "sedE_kwargs": self._sedE.defaults,
            "cl_kwargs": self._cl.defaults,
        }

    def _get_repr(self):
        sedT_repr = self._sedT._get_repr()
        key = list(sedT_repr.keys())[0]
        sedT_repr[key + " (SED T)"] = sedT_repr.pop(key)

        sedE_repr = self._sedE._get_repr()
        key = list(sedE_repr.keys())[0]
        sedE_repr[key + " (SED E)"] = sedE_repr.pop(key)

        cl_repr = self._cl._get_repr()
        key = list(cl_repr.keys())[0]
        cl_repr[key + " (Cl)"] = cl_repr.pop(key)

        return {type(self).__name__: [sedT_repr, sedE_repr, cl_repr]}

    def eval(self, sedT_kwargs={}, sedE_kwargs={}, cl_kwargs={}):
        """Compute the model at frequency and ell combinations.

        Parameters
        ----------
        sedT_args : list
            Arguments for which the temperature `sed` is evaluated.
        sedE_args : list
            Arguments for which the E pol `sed` is evaluated.
        cl_args : list
            Arguments for which the `cl` is evaluated.

        Returns
        -------
        cross : ndarray
            Cross-spectrum. The shape is ``(..., freq, freq, ell)``.
        """
        fT_nu = self._sedT(**sedT_kwargs)
        fE_nu = self._sedE(**sedE_kwargs)
        cl = self._cl(**cl_kwargs)
        # f_nu.shape can be either [freq] or [ell, freq]
        if fT_nu.shape[0] != cl.shape[-1] or (
            fT_nu.shape[0] == 1 and cl.shape[-1] == 1
        ):
            fT_nu = fT_nu[np.newaxis]
        if fE_nu.shape[0] != cl.shape[-1] or (
            fE_nu.shape[0] == 1 and cl.shape[-1] == 1
        ):
            fE_nu = fE_nu[np.newaxis]

        return np.einsum("l...i,l...j,...l->...ijl", fT_nu, fE_nu, cl)


class DecorrelatedFactorizedCrossSpectrum(Model):
    r"""Decorrelated factorized cross spectrum

    Cross-spectrum of **one** component for which the scaling in frequency
    and in multipoles are factorizable. The scaling in frequency is not exact
    and some decorrelation is specified.

    .. math:: xC_{\ell}^{(ij)} = f_{decor}(\nu_i, \nu_j) f(\nu_j) f(\nu_i) C_{\ell}

    This model implements a decorrelation by rigidely rescalling a decorelation ratio
    with respect to a reference cross-term. For example, for 3 frequencies, with the
    reference taken at $\nu_1\times\nu_3$:

    .. math::  f_{decor} = I(3) + \Delta_{decor} \begin{pmatrix}
        0 & r_{12} & 1 \\
        r_{12} & 0 & r_{23} \\
        1 & r_{23} & 0
        \end{pmatrix}

    Parameters
    ----------
    sed : callable
        :math:`f(\nu)`. It returns an array with shape ``(..., freq)``.
        It can be :class:`fgspectra.frequency.SED`.
    cl : callable
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
        kw_decor = {
            key: kwargs[key]
            for key in kwargs.keys()
            if key not in ["sed_kwargs", "cl_kwargs"]
        }
        super().set_defaults(**kw_decor)

    @property
    def defaults(self):
        kw_decor = {
            key: super().defaults[key]
            for key in super().defaults.keys()
            if key not in ["kwargs"]
        }
        return {
            **kw_decor,
            "sed_kwargs": self._sed.defaults,
            "cl_kwargs": self._cl.defaults,
        }

    def _get_repr(self):
        decor_repr = super()._get_repr()
        key = list(decor_repr.keys())[0]
        decor_repr[key] = {
            k: decor_repr[key][k]
            for k in decor_repr[key].keys()
            if k not in ["sed_kwargs", "cl_kwargs"]
        }
        decor_repr["Decorrelation"] = decor_repr.pop(key)

        sed_repr = self._sed._get_repr()
        key = list(sed_repr.keys())[0]
        sed_repr[key + " (SED)"] = sed_repr.pop(key)

        cl_repr = self._cl._get_repr()
        key = list(cl_repr.keys())[0]
        cl_repr[key + " (Cl)"] = cl_repr.pop(key)

        return {type(self).__name__: [decor_repr, sed_repr, cl_repr]}

    def eval(self, decor=None, f_decor=None, sed_kwargs={}, cl_kwargs={}):
        """Compute the model at frequency and ell combinations.

        Parameters
        ----------
        sed_args : list
            Arguments for which the `sed` is evaluated.
        cl_args : list
            Arguments for which the `cl` is evaluated.
        decor : float
            Decorelation facot, by which f_decor is rescaled
        f_decor : ndarray
            Array of the decorelation scaling in frequency.
            Shape is ``(freq, freq)``.
        Returns
        -------
        cross : ndarray
            Cross-spectrum. The shape is ``(..., freq, freq, ell)``.
        """

        decorrelation = np.eye(f_decor.shape[0]) + decor * f_decor

        f_nu = self._sed(**sed_kwargs)
        cl = self._cl(**cl_kwargs)
        if f_nu.shape[0] != cl.shape[-1] or (f_nu.shape[0] == 1 and cl.shape[-1] == 1):
            f_nu = f_nu[np.newaxis]

        res = np.einsum("ij,l...i,l...j,...l->...ijl", decorrelation, f_nu, f_nu, cl)

        return res


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
        # verifying that sed.shape[1] is ell, otherwise adding newaxis
        if f_nu.shape[1] != cl.shape[-1] or (f_nu.shape[1] == 1 and cl.shape[-1] == 1):
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
