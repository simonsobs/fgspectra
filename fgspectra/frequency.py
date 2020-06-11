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
from scipy.interpolate  import interp1d
from scipy.integrate import quad
import itertools

import glob

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



class RadioSourcesFluxCut():
    """ Radio point source power law with amplitude from flux cut

    .. math:: f(\nu) = amp (\nu / \nu_0)^{\beta}
    .. math:: amp =
    """

    def get_number_counts(self,nu  ):
        """
        Given the frequency channels finds the total intensity  number counts from predictions of  Tucci et al. 2011 model and computes the
        polarization number counts as in :func:`get_differential_number_counts`.
        """

        model_avail= glob.glob(  '../fgspectra/data/lagache_number_counts/ns*_radio.dat')
        frequency_model_avail=[(k.split('ns')[1].split('_')[0]) for k in model_avail]
        idxfreq=np.argmin(abs(nu  - np.array(list(map(float,frequency_model_avail)) ) ) )
        lagachemodels= np.loadtxt(model_avail[idxfreq])
        S= lagachemodels [:,0]
        differential_counts  =lagachemodels[:,1]

        dnds=interp1d(S, differential_counts )
        return S,dnds

    def brightness2Kcmb(self, nu):
        """
        Returns conversion factor to pass from ``Ib``  brightness units
        to physical temperature at a given frequency ``nu``
        see eq.(7) in Puglisi et al.2018,
        http://stacks.iop.org/0004-637X/858/i=2/a=85?key=crossref.8fabfadab2badba8bccaabda16342429
        """
        nu0 = 56.8 # * u.gigahertz

        x = nu / nu0
        return ( 4.0e-2 * (np.exp(x) - 1) ** 2 / (x ** 4 * np.exp(x)) ) # u.uK / (u.Jy / u.sr))


    def estimate_auto_spectra(self):
        """
        Given differential  number counts :math:`dN/dS`, estimated at certain  flux densities :math:`S\in [0, S_{cut}]`,
        it estimates the integral

        .. math::

            C^{radio}_{\nu} (S_{cut} )= \int _{0} ^{ S_{cut}} dS {n(S)} S^2

        the output of this estimate is a dictionary  of values  of :math:`C^{radio}_{\nu} (S_{cut} )` at a given frequency and Scut .
        """
        auto_spectra = {}
        for i,nu in enumerate (self.nu ) :
            S, dndS = self.get_number_counts( nu )
            Integral = self.integrate_number_counts(S,dndS, self.fluxcut[i] )
            flux2Kcmb = self.brightness2Kcmb(nu)
            auto_spectra[nu] = flux2Kcmb ** 2 * Integral

        return auto_spectra

    def integrate_number_counts (self, S, dndS , Scut ):
        """
        integral from 0 to Scut

        """
        # function to integrate

        integrand = lambda s: dndS (s) * s ** 2
        Smin=  S.min()
        # integral N(<S) per srad
        Integral = (
            quad(integrand, Smin, Scut  , limit=1000, epsrel=1.0e-3)[0]

        )
        return Integral


    def get_SED_from_Planck (self ):

        data = np.load('../fgspectra/data/PointSource_Spectral_indices_Planck_catalogue.npy')
        nu        = data[:,0]
        alpha_bar = data[:,1]
        sigma     = data[:,2]

        self.alpha_bar = interp1d(nu,alpha_bar,fill_value='extrapolate', )
        self.sigma     = interp1d(nu,sigma,    fill_value='extrapolate',  )

    def estimate_cross_spectra ( self ,  nu0 =100  ):
        """
        implementing the Xspectra approach in Reichardt et al. 2012  (R12, see eq. 18 )

        .. math::

            D^{radio}_{\nu1, \nu2}(S_{cut1}, S_{cut2}) = D^{radio}_{\ell=3000, \nu0} \left( \frac{\ell}{3000}\right)^2 \epsilon_{\nu1,\nu2} \eta_{\nu1,\nu2} ^{\alpha +0.5 \ln(\eta_{\nu1,\nu2}) \sigma^2  }

        The major difference is in the estimates of the distribution of spectral indices for
        point  sources obtained from the Planck Catalogue PCCS2 .
        The R12 approach rely on a reference freq.,  nu0, and a fluxcut related to it in order
        to forecast the cross power.
        the default for nu0 is set to 100 GHz
        whereas the flux cut for each cross power is set to the maximum value between the
        flux cuts at the two frequency, this is motivated by the fact that the poissonian contribution
        to the power spectra is nearly proportional to Scut.

        the output of this estimate is a dictionary  of amplitude of  (:math:`D^{radio}_{\ell=3000,\nu1, \nu2}(S_{cut1}, S_{cut2})`)
         with keys the  tuple of (nu1, nu2)  combinations.

        """
        Dcross ={}
        ell0 =3000
        for nu1,nu2 in itertools.combinations(self.nu, 2 ):
            id1 = np.argmin(np.fabs (self.nu -nu1))
            id2 = np.argmin(np.fabs (self.nu -nu2))
            fluxcut_nu0  = np.max ((self.fluxcut[id1], self.fluxcut[id2]))
            eta = (nu1*nu2) / nu0**2
            epsilon  = ( self.brightness2Kcmb(nu0) *  self.brightness2Kcmb(nu0)
                                                    /
                        (self.brightness2Kcmb(nu1)* self.brightness2Kcmb(nu2) ) )  # see eq.14 of R12

            S, dnds = self.get_number_counts( nu0 )
            Integral_counts  = self.integrate_number_counts(S,dnds, fluxcut_nu0 )
            flux2Kcmb = self.brightness2Kcmb(nu0)
            D3000_nu0 = flux2Kcmb *Integral_counts *ell0 *(ell0+1 )/2/np.pi
            self.get_SED_from_Planck()
            eta_exponent = (self.alpha_bar(nu0) +0.5* np.log(eta)*self.sigma(nu0 )**2 )
            #print(  D3000_nu0 , epsilon , eta , eta_exponent )
            Dcross[(nu1,nu2)]=  D3000_nu0 * epsilon * eta ** eta_exponent

        return Dcross


    def __init__ (self, nu=None,   fluxcut=None  ):
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
        fwhm: float
            beam FWHM in arcmin

        Returns
        -------
        sed: ndarray
            The last dimension is the frequency dependence.
            The leading dimensions are the broadcast between the hypothetic
            dimensions of `beta` and `fluxcut`.
        """
        self.fluxcut= fluxcut
        self.nu = nu
        try:
            assert type(self.nu)==list or type(self.nu)==np.array
            assert type(self.fluxcut )== list or type(self.fluxcut )== np.array
        except AssertionError :
            self.nu =  [self.nu]
            self.fluxcut= [self.fluxcut]
        finally :
            self.fluxcut= np.array(self.fluxcut )
            self.nu=np.array(self.nu)






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
        amp = np.array(amp)
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
