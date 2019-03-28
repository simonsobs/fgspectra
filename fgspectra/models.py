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
        f_nu = self._cl(*sed_args)[..., np.newaxis]
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
        return np.einsum('k...i,n...j,...knl->...ijl',
                         f_nu, f_nu, self._cl(*cl_args))


class PowerLaw(FactorizedCrossSpectrum):
    """ Single Power law in both fequency and multipoles

    See :class:`fgspectra.frequency.PowerLaw` for the frequency dependence and
    :class:`fgspectra.power.PowerLaw` for the ell-dependence.
    """

    def __init__(self):
        super().__init__(fgf.PowerLaw(), fgp.PowerLaw())


'''
class ThermalSZ(CompositeModel):
    """Model for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self, a_tSZ=None,
                 nu_0=150.0, ell_0=3000):
        """Initialize the ThermalSZ object with required parameters."""

        self.params = { 'nu_0' : nu_0,
                        'ell_0' : ell_0,
                        'a_tSZ' : a_tSZ}

        self.fnu = fgspectra.frequency.ThermalSZFreq() # f(nu) template
        self.Dl_tSZ = fgspectra.power.tSZ_150_bat() # D_ell template

        return

    def model(self, nu_i, nu_j, ell, **kwargs):
        """Compute the Thermal SZ at frequency and ell combinations.

        Parameters
        ----------
        nu_i : float
            first frequency channel in cross-spectrum
        nu_j : float
            second frequency channel in cross-spectrum
        ells : 1D array
            ells at which we evaluate the model
        a_tSZ : float
            overall amplitude of the tSZ model

        Returns
        -------
        1D array of floats : model result of same shape as input ells
        """
        par = self.get_missing(kwargs)
        return par['a_tSZ'] * (
            self.fnu(nu_i) * self.fnu(nu_j) / self.fnu(par['nu_0'])**2 *
            self.Dl_tSZ(ell) / self.Dl_tSZ(par['ell_0']) )


class KinematicSZ(CompositeModel):
    """Model for Kinematic Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self, a_kSZ=None,
                 nu_0=150.0, ell_0=3000):
        """Initialize the KinematicSZ object with required parameters."""

        self.params = { 'nu_0' : nu_0,
                        'ell_0' : ell_0,
                        'a_kSZ' : a_kSZ}

        self.Dl_kSZ = fgspectra.power.kSZ_bat() # D_ell template

        return

    def model(self, nu_i, nu_j, ell, **kwargs):
        """Compute the KinematicSZ SZ at frequency and ell combinations.

        Parameters
        ----------
        nu_i : float
            first frequency channel in cross-spectrum
        nu_j : float
            second frequency channel in cross-spectrum
        ells : 1D array
            ells at which we evaluate the model
        a_kSZ : float
            overall amplitude of the tSZ model

        Returns
        -------
        1D array of floats : model result of same shape as input ells
        """
        par = self.get_missing(kwargs)
        return par['a_kSZ'] * (
            self.Dl_kSZ(ell) / self.Dl_kSZ(par['ell_0']) )


class CIBP(CompositeModel):
    """Model for CIB Poisson component (Dunkley et al. 2013)."""

    def __init__(self, a_p=None, beta_p=None,
                 T_d=9.7, T_CMB=2.725, nu_0=150.0, ell_0=3000):
        """Initialize the CIB object with required parameters."""

        self.params = { 'nu_0' : nu_0,
                        'ell_0' : ell_0,
                        'a_p' : a_p,
                        'beta_p' : beta_p,
                        'T_d' : T_d,
                        'T_CMB' : T_CMB}

        self.mu = fgspectra.frequency.CIBFreq()
        return

    def model(self, nu_i, nu_j, ell, **kwargs):
        """Compute the Thermal SZ at frequency and ell combinations.

        Parameters
        ----------
        nu_i : float
            first frequency channel in cross-spectrum
        nu_j : float
            second frequency channel in cross-spectrum
        ells : 1D array
            ells at which we evaluate the model
        a_tSZ : float
            overall amplitude of the tSZ model

        Returns
        -------
        1D array of floats : model result of same shape as input ells
        """
        par = self.get_missing(kwargs)

        mu_i = self.mu(nu_i, beta=par['beta_p'],
                      T_d=par['T_d'], T_CMB=par['T_CMB'])
        mu_j = self.mu(nu_j, beta=par['beta_p'],
                      T_d=par['T_d'], T_CMB=par['T_CMB'])
        mu0 = self.mu(par['nu_0'], beta=par['beta_p'],
                      T_d=par['T_d'], T_CMB=par['T_CMB'])
        return par['a_p'] * (ell / par['ell_0'])**2 * (mu_i * mu_j / mu0**2)


class CIBC(CompositeModel):
    """Model for CIB clustering component (Dunkley et al. 2013)."""

    def __init__(self, a_c=None, beta_c=None, n_CIBC=None,
                 T_d=9.7, T_CMB=2.725, nu_0=150.0, ell_0=3000):
        """Initialize the CIB object with required parameters."""

        self.params = { 'nu_0' : nu_0,
                        'ell_0' : ell_0,
                        'a_c' : a_c,
                        'beta_c' : beta_c,
                        'n_CIBC' : n_CIBC,
                        'T_d' : T_d,
                        'T_CMB' : T_CMB}

        self.mu = fgspectra.frequency.CIBFreq()
        return

    def model(self, nu_i, nu_j, ell, **kwargs):
        """Compute the Thermal SZ at frequency and ell combinations.

        Parameters
        ----------
        nu_i : float
            first frequency channel in cross-spectrum
        nu_j : float
            second frequency channel in cross-spectrum
        ells : 1D array
            ells at which we evaluate the model
        a_tSZ : float
            overall amplitude of the tSZ model

        Returns
        -------
        1D array of floats : model result of same shape as input ells
        """
        par = self.get_missing(kwargs)
        mu_i = self.mu(nu_i, beta=par['beta_c'],
                      T_d=par['T_d'], T_CMB=par['T_CMB'])
        mu_j = self.mu(nu_j, beta=par['beta_c'],
                      T_d=par['T_d'], T_CMB=par['T_CMB'])
        mu_0 = self.mu(par['nu_0'], beta=par['beta_c'],
                      T_d=par['T_d'], T_CMB=par['T_CMB'])

        from scipy import constants
        x = constants.h * (nu_i*1e9) / (constants.k * par['T_CMB'])
        nu_0 = par['nu_0']
        x0 = constants.h * (nu_0*1e9) / (constants.k * par['T_CMB'])
        T_CMB = par['T_CMB']
        T_d = par['T_d']
        top =  (np.cosh(x) - 1) / (
              (nu_i*1e9)**4)
        bottom =  (np.cosh(x0) - 1) / (
              (nu_0*1e9)**4)



        return par['a_c'] * (ell*(ell+1)/3000**2)*(ell/3000)**(-par['n_CIBC']
            ) * (mu_i * mu_j / mu_0**2)

# %%

class tSZxCIB(CompositeModel):
    """Model for tSZxCIB clustering component (Dunkley et al. 2013)."""

    def __init__(self, xi=None, a_c=None, beta_c=None, a_tSZ=None,
                 T_d=9.7, T_CMB=2.725, nu_0=150.0, ell_0=3000):
        """Initialize the tSZxCIB object with required parameters."""

        self.params = {'nu_0' : nu_0,
                       'ell_0' : ell_0,
                       'xi' : xi,
                       'a_c' : a_c,
                       'beta_c' : beta_c,
                       'a_tSZ' : a_tSZ,
                       'T_d' : T_d,
                       'T_CMB' : T_CMB}

        self.mu = fgspectra.frequency.CIBFreq() # mu template for CIB
        self.fnu = fgspectra.frequency.ThermalSZFreq() # f(nu) template tSZ
        self.Dl_tSZxCIB = fgspectra.power.sz_x_cib_template()
        return

    def model(self, nu_i, nu_j, ell, **kwargs):
        """Compute the Thermal SZ at frequency and ell combinations.

        Parameters
        ----------
        nu_i : float
            first frequency channel in cross-spectrum
        nu_j : float
            second frequency channel in cross-spectrum
        ells : 1D array
            ells at which we evaluate the model
        a_tSZ : float
            overall amplitude of the tSZ model

        Returns
        -------
        1D array of floats : model result of same shape as input ells
        """
        par = self.get_missing(kwargs)
        mu_i = self.mu(nu_i, beta=par['beta_c'],
                      T_d=par['T_d'], T_CMB=par['T_CMB'])
        mu_j = self.mu(nu_j, beta=par['beta_c'],
                      T_d=par['T_d'], T_CMB=par['T_CMB'])
        mu_0 = self.mu(par['nu_0'], beta=par['beta_c'],
                      T_d=par['T_d'], T_CMB=par['T_CMB'])
        
        # hack to make effective frequency
        if kwargs['effective_frequencies']:
            i, j = kwargs['i'], kwargs['j']
            sz_nu_i = kwargs[f'f{i}_sz']
            sz_nu_j = kwargs[f'f{j}_sz']
            fp = self.fnu(sz_nu_i) * mu_j + self.fnu(sz_nu_j) * mu_i
        else:
            fp = self.fnu(nu_i) * mu_j + self.fnu(nu_j) * mu_i
        fp_0 = self.fnu(par['nu_0']) * mu_0 * 2

        return -par['xi'] * np.sqrt(par['a_tSZ'] * par['a_c']) * (2 *
            fp / fp_0 * self.Dl_tSZxCIB(ell) / self.Dl_tSZxCIB(par['ell_0']))


class RadioPointSources(CompositeModel):
    """Model for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self, a_s=None,
                 alpha_s=-0.5, nu_0=150.0, ell_0=3000, T_CMB=2.725):
        """Initialize the ThermalSZ object with required parameters."""

        self.params = {'a_s' : a_s,
                       'alpha_s' : alpha_s,
                       'nu_0' : nu_0,
                       'ell_0' : ell_0,
                       'T_CMB' : T_CMB}

        self.cib = fgspectra.frequency.CIBFreq() # mu template for CIB
        return

    def model(self, nu_i, nu_j, ell, **kwargs):
        """Compute the Thermal SZ at frequency and ell combinations.

        Parameters
        ----------
        nu_i : float
            first frequency channel in cross-spectrum
        nu_j : float
            second frequency channel in cross-spectrum
        ells : 1D array
            ells at which we evaluate the model
        a_tSZ : float
            overall amplitude of the tSZ model

        Returns
        -------
        1D array of floats : model result of same shape as input ells
        """
        par = self.get_missing(kwargs)
        return par['a_s'] * (ell / par['ell_0'])**2 * ( nu_i * nu_j /
            par['nu_0']**2)**par['alpha_s'] * (self.cib.g(nu_i, par['T_CMB']) *
            self.cib.g(nu_j, par['T_CMB']) /
            self.cib.g(par['nu_0'], par['T_CMB'])**2 )


class GalacticCirrus(CompositeModel):
    """Model for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self, a_g=None,
                 beta_g=3.8, n_g=-0.7,
                 nu_0=150.0, ell_0=3000, T_CMB=2.725):
        """Initialize the ThermalSZ object with required parameters."""

        self.params = {'a_g' : a_g,
                       'beta_g' : beta_g,
                       'n_g' : n_g,
                       'nu_0' : nu_0,
                       'ell_0' : ell_0,
                       'T_CMB' : T_CMB}

        self.cib = fgspectra.frequency.CIBFreq() # mu template for CIB
        return

    def model(self, nu_i, nu_j, ell, **kwargs):
        """Compute the Thermal SZ at frequency and ell combinations.

        Parameters
        ----------
        nu_i : float
            first frequency channel in cross-spectrum
        nu_j : float
            second frequency channel in cross-spectrum
        ells : 1D array
            ells at which we evaluate the model
        a_tSZ : float
            overall amplitude of the tSZ model

        Returns
        -------
        1D array of floats : model result of same shape as input ells
        """
        par = self.get_missing(kwargs)
        return par['a_g'] * (ell / par['ell_0'])**par['n_g'] * ( nu_i * nu_j /
            par['nu_0']**2)**par['beta_g'] * (self.cib.g(nu_i, par['T_CMB']) *
            self.cib.g(nu_j, par['T_CMB']) /
            self.cib.g(par['nu_0'], par['T_CMB'])**2 )
'''
