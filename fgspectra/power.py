r"""
Power spectrum

This module implements the ell-dependent component of common foreground
contaminants.

This module draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).
"""

import os
import pkg_resources
import numpy as np
from .model import Model


def _get_power_file(model):
    """ File path for the named model
    """
    data_path = pkg_resources.resource_filename('fgspectra', 'data/')
    filename = os.path.join(data_path, 'cl_%s.dat'%model)
    if os.path.exists(filename):
        return filename
    raise ValueError('No template for model '+model)


class PowerSpectrumFromFile(Model):
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

    def __init__(self, filenames, **kwargs):
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
                pad_width = [(0, 0)] * self._cl.ndim
                pad_width[-1] = (0, n_missing_ells)
                self._cl = np.pad(self._cl, pad_width,
                                  mode='constant', constant_values=0)

            self._cl[i+(ell,)] = spec
        self.set_defaults(**kwargs)

    def eval(self, ell=None, ell_0=None, amp=1.0):
        """Compute the power spectrum with the given ell and parameters."""
        amp = np.array(amp)[..., np.newaxis]
        return amp * self._cl[..., ell] / self._cl[..., ell_0]

class PowerSpectrumFromFile_PowerLaw(PowerSpectrumFromFile):

    def eval(self, ell=None, ell_0=None, alpha=None, amp=1.0):
        amp = np.array(amp)[..., np.newaxis]
        alpha = np.array(alpha)[..., np.newaxis]
        res = self._cl[..., ell] / self._cl[..., ell_0]
        res[..., ell > ell_0] = (ell[ell > ell_0]/ell_0)**alpha
        return amp*res


class tSZ_150_bat(PowerSpectrumFromFile):
    """PowerSpectrum for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self):
        """Intialize object with parameters."""
        super().__init__(_get_power_file('tsz_150_bat'))


class kSZ_bat(PowerSpectrumFromFile):
    """PowerSpectrum for Kinematic Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self):
        """Intialize object with parameters."""
        super().__init__(_get_power_file('ksz_bat'))


class PowerLaw(Model):
    r""" Power law

    .. math:: C_\ell = (\ell / \ell_0)^\alpha
    """
    def eval(self, ell=None, alpha=None, ell_0=None, amp=1.0):
        """

        Parameters
        ----------
        ell: float or array
            Multipole
        alpha: float or array
            Spectral index.
        ell_0: float
            Reference ell
        amp: float or array
            Amplitude, shape must be compatible with `alpha`.

        Returns
        -------
        cl: ndarray
            The last dimension is ell.
            The leading dimensions are the hypothetic dimensions of `alpha`
        """
        alpha = np.array(alpha)[..., np.newaxis]
        amp = np.array(amp)[..., np.newaxis]
        return amp * (ell / ell_0)**alpha


class CorrelatedPowerLaws(PowerLaw):
    """ As PowerLaw, but requires only the diagonal of the component-component
    dimensions and the correlation coefficient between TWO PL
    """
    def eval(self, ell=None, alpha=None, ell_0=None, amp=None, rho=None):
        """
        Parameters
        ----------
        ell: float or array
            Multipole
        amp: array
            amplitude of the auto-spectra. Shape must be ``(..., 2)``, which
            means that for time being we support only two correlated components
        alpha: float or array
            Spectral index. Shape must be broadcast-compatible with `amp`
        ell_0: float
            Reference ell

        Returns
        -------
        cl: ndarray
            The dimensions are ``(..., comp1, comp2, ell)``.
            The leading dimensions are the hypothetic dimensions of `alpha` and
            `amp`.
        """
        alpha = np.array(alpha)
        amp = np.array(amp)

        alpha = (alpha[..., np.newaxis, :] + alpha[..., np.newaxis]) / 2
        amp = (amp[..., np.newaxis, :]
               * amp[..., np.newaxis])**0.5
        amp[..., 1, 0] *= rho
        amp[..., 0, 1] *= rho

        return super().eval(ell=ell, alpha=alpha, ell_0=ell_0, amp=amp)



class PowerSpectraAndCorrelation(Model):
    r"""Components' spectra and their correlation

    Spectrum of correlated components defined by the spectrum of each component
    and their correlation

    Parameters
    ----------
    *power_spectra : series of `PowerSpectrum`
        The series has lenght :math:`N (N + 1) / 2`, where :math:`N` is the
        number of components. They specify the upper (or lower) triangle of the
        component-component cross spectra, which is symmetric. The series stores
        a `PowerSpectrum` for each component-component combination.
        The main diagonal (i.e. the autospectra) goes first, the second diagonal
        of the correlation matrix follows, then the third, etc.
        The ordering is similar to the one returned by `healpy.anafast`.
    """


    def __init__(self, *power_spectra, **kwargs):
        self._power_spectra = power_spectra
        self.n_comp = np.rint(-1 + np.sqrt(1 + 8 * len(power_spectra))) // 2
        self.n_comp = int(self.n_comp)
        assert (self.n_comp + 1) * self.n_comp // 2 == len(power_spectra)
        self.set_defaults(**kwargs)

    def set_defaults(self, **kwargs):
        if 'kwseq' in kwargs:
            for i, cl_kwargs in enumerate(kwargs['kwseq']):
                self._power_spectra[i].set_defaults(**cl_kwargs)

    @property
    def defaults(self):
        return {'kwseq': [ps.defaults for ps in self.power_spectra]}

    def _get_repr(self):
        return {type(self).__name__:
                    [ps._get_repr() for ps in self.power_spectra]}


    def eval(self, kwseq=None):
        """Compute the SED with the given frequency and parameters.
        
        Parameters
        ----------
        *argss
            The length of `argss` has to be equal to the number of SEDs joined.
            ``argss[i]`` is the argument list of the ``i``-th SED.
        
        """
        spectra = [ps(*args) for ps, args in zip(self._power_spectra, argss)]
        corrs = spectra[self.n_comp:]
        cls = spectra[:self.n_comp]
        sqrt_cls = [np.sqrt(cl) for cl in cls]
        cls_shape = np.broadcast(*cls).shape

        res = np.empty(  # Shape is (..., comp, comp, ell)
            cls_shape[1:-1] + (self.n_comp, self.n_comp) + cls_shape[-1:])

        for i in range(self.n_comp):
            res[..., i, i, :] = cls[i]

        i_corr = 0
        for k_off_diag in range(1, self.n_comp):
            for el_off_diag in range(self.n_comp - k_off_diag):
                i = el_off_diag
                j = el_off_diag + k_off_diag
                res[..., i, j, :] = sqrt_cls[i] * sqrt_cls[j] * corrs[i_corr]
                res[..., j, i, :] = res[..., i, j, :]
                i_corr += 1

        '''
        i_corr = 0
        for i in range(self.n_comp):
            res[..., i, i, :] = cls[i]
            for j in range(i + 1, self.n_comp):
                res[..., i, j, :] = sqrt_cls[i] * sqrt_cls[j] * corrs[i_corr]
                res[..., j, i, :] = res[..., i, j, :]
                i_corr += 1
        '''

        assert i_corr == len(corrs)
        return res

class PowerSpectraAndCovariance(Model):
    r"""Components' spectra and their covariance

    Spectrum of correlated components defined by the spectrum of each component
    and their correlation

    Parameters
    ----------
    *power_spectra : series of `PowerSpectrum`
        The series has length :math:`N (N + 1) / 2`, where :math:`N` is the
        number of components. They specify the upper (or lower) triangle of the
        component-component cross spectra, which is symmetric. The series stores
        a `PowerSpectrum` for each component-component combination.
        The main diagonal (i.e. the autospectra) goes first, the second diagonal
        of the covariance matrix follows, then the third, etc.
        The ordering is similar to the one returned by `healpy.anafast`.
    """

    def __init__(self, *power_spectra, **kwargs):
        self._power_spectra = power_spectra
        self.n_comp = np.rint(-1 + np.sqrt(1 + 8 * len(power_spectra))) // 2
        self.n_comp = int(self.n_comp)
        assert (self.n_comp + 1) * self.n_comp // 2 == len(power_spectra)
        self.set_defaults(**kwargs)

    def set_defaults(self, **kwargs):
        if 'kwseq' in kwargs:
            for i, cl_kwargs in enumerate(kwargs['kwseq']):
                self._power_spectra[i].set_defaults(**cl_kwargs)

    @property
    def defaults(self):
        return {'kwseq': [ps.defaults for ps in self._power_spectra]}

    def _get_repr(self):
        return {type(self).__name__:
                    [ps._get_repr() for ps in self._power_spectra]}

    def eval(self, kwseq=None):
        """Compute the Cl with the given frequency and parameters.

        kwseq
            The length of `argss` has to be equal to the number of SEDs joined.
            ``kwseq[i]`` is the argument list of the ``i``-th SED.
        """
        spectra = np.array(
            [ps(**kwargs) for ps, kwargs in zip(self._power_spectra, kwseq)])
        res = np.empty(  # Shape is (..., comp, comp, ell)
            spectra.shape[1:-1] + (self.n_comp, self.n_comp) + spectra.shape[-1:])
        
        i_corr = 0
        for k_off_diag in range(0, self.n_comp):
            for el_off_diag in range(self.n_comp - k_off_diag):
                i = el_off_diag
                j = el_off_diag + k_off_diag
                res[..., i, j, :] = spectra[i_corr]
                res[..., j, i, :] = res[..., i, j, :]
                i_corr += 1

        assert i_corr == len(spectra)
        return res


class SZxCIB_Reichardt2012(PowerSpectraAndCorrelation):
    """PowerSpectrum for SZxCIB (Dunkley et al. 2013)."""

    def __init__(self, **kwargs):
        """Intialize object with parameters."""
        power_spectra = [
            PowerSpectrumFromFile(_get_power_file('tsz_150_bat')),
            PowerLaw(),
            lambda xi : xi 
        ]
        super().__init__(*power_spectra)
        
        
class SZxCIB_Addison2012(PowerSpectraAndCovariance):
    """PowerSpectrum for SZxCIB (Dunkley et al. 2013)."""

    def __init__(self, **kwargs):
        """Intialize object with parameters."""
        power_spectra = [
            PowerSpectrumFromFile(_get_power_file('tsz_150_bat')),
            PowerLaw(),
            PowerSpectrumFromFile(_get_power_file('sz_x_cib'))
        ]
        super().__init__(*power_spectra, **kwargs)

class SquarePowerLaw(Model):
    def eval(self, ell=None, ell_0=None, amp=1.0):
        """
        Parameters
        ----------
        ell: float or array
            Multipole
        ell_0: float
            Reference ell
        amp: float or array
            Amplitude, shape must be compatible with `ell`.

        Returns
        -------
        cl: ndarray
            Has same shape as ell.
        """
        amp = np.array(amp)[..., np.newaxis]
        return amp * (ell * (ell + 1.0)) / (ell_0 * (ell_0 + 1.0))



class CIB_Planck(Model):
    """Planck CIB template.
       HTJ - after plik_v22 FORTRAN

      Do not use this. It is hacked together because the Planck template is
      not really designed to work well within fgspectra."""

    # def __init__(self, **kwargs):
    #     #     spec = np.genfromtxt(_get_power_file('cib_planck'), unpack=False, dtype=np.float)
    #     #     ell = spec[:, 0].astype(int)
    #     #     self._cl = np.zeros((max(ell) + 1, 4))
    #     #     self._cl[ell, 0] = spec[:, 1] * (4096.68168783 / 1e6) ** 2.0
    #     #     self._cl[ell, 1] = spec[:, 7] * (2690.05218701 / 1e6) ** 2.0
    #     #     self._cl[ell, 2] = spec[:, 8] * (2690.05218701 / 1e6) * (2067.43988919 / 1e6)
    #     #     self._cl[ell, 3] = spec[:, 12] * (2067.43988919 / 1e6) ** 2.0
    #     #
    #     #     ls = np.arange(self._cl.shape[0])[..., np.newaxis]
    #     #     norm = self._cl[3000, 3]
    #     #     self._cl = (self._cl / norm) * ls * (ls + 1.0) / (3000.0 * 3001.0)
    #     #
    #     #     self.set_defaults(**kwargs)
    #     #
    #     # def eval(self, ell=None, ell_0=None, n_cib=None, amp=1.0):
    #     #     """Compute the power spectrum with the given ell and parameters."""
    #     #     if np.isscalar(ell): ell = np.array(ell)[..., np.newaxis]
    #     #
    #     #     return amp * self._cl[ell, :] * (ell[:, np.newaxis] / ell_0) ** (n_cib + 1.3

    def __init__(self, **kwargs):
        spec = np.genfromtxt(_get_power_file('cib_planck'), unpack=False, dtype=float)
        ell = spec[:, 0].astype(int)
        self._cl = np.zeros((3, 3, max(ell) + 1))
        self._cl[0, 0, ell] = spec[:, 1] * (4096.68168783 / 1e6) ** 2.0
        self._cl[1, 1, ell] = spec[:, 7] * (2690.05218701 / 1e6) ** 2.0
        self._cl[1, 2, ell] = spec[:, 8] * (2690.05218701 / 1e6) * (2067.43988919 / 1e6)
        self._cl[2, 1, ell] = spec[:, 8] * (2690.05218701 / 1e6) * (2067.43988919 / 1e6)
        self._cl[2, 2, ell] = spec[:, 12] * (2067.43988919 / 1e6) ** 2.0

        ls = np.arange(self._cl.shape[-1])[np.newaxis, np.newaxis, :]
        norm = self._cl[2, 2, 3000]
        self._cl = (self._cl / norm) * ls * (ls + 1.0) / (3000.0 * 3001.0)
        self.set_defaults(**kwargs)

    def eval(self, ell=None, ell_0=None, n_cib=None, amp=1.0):
        """Compute the power spectrum with the given ell and parameters."""
        if np.isscalar(ell): ell = np.array(ell)[..., np.newaxis]

        return amp * self._cl[..., ell] * (ell / ell_0) ** (n_cib + 1.3)


# class gal_Planck(Model):
#     """Planck gal template.
#        HTJ - after plik_v22 FORTRAN
#
#       Do not use this. It is hacked together because the Planck template is
#       not really designed to work well within fgspectra."""
#
#     def __init__(self, **kwargs):
#         self._cl = np.zeros((2601, 4))
#         for i, filename in enumerate(['gal_planck_100', 'gal_planck_143', 'gal_planck_143x217', 'gal_planck_217']):
#             ell, spec, _ = np.genfromtxt(_get_power_file(filename), unpack=True)
#
#             self._cl[ell.astype(int), i] = spec * ell * (ell + 1.0) / (2.0 * np.pi)
#             self._cl[:, i] /= self._cl[200, i]
#
#         self.set_defaults(**kwargs)
#
#     def eval(self, ell=None):
#         """Compute the power spectrum with the given ell and parameters."""
#         if np.isscalar(ell):
#             ell = np.array(ell)[..., np.newaxis]
#
#         res = np.tile(np.nan, ell.shape + (4,))
#         res[ell <= 2600] = self._cl[ell[ell <= 2600], :]
#         return res
class gal_Planck(Model):
    def __init__(self, **kwargs):
        self._cl = np.zeros((3,3,2601))
        #100x100
        ell, spec, _ = np.genfromtxt(_get_power_file('gal_planck_100'), unpack=True)
        self._cl[0,0,ell.astype(int)] = spec * ell * (ell + 1.0) / (2.0 * np.pi)
        self._cl[0,0] /= self._cl[0,0,200]
        #143x143
        ell, spec, _ = np.genfromtxt(_get_power_file('gal_planck_143'), unpack=True)
        self._cl[1,1,ell.astype(int)] = spec * ell * (ell + 1.0) / (2.0 * np.pi)
        self._cl[1,1] /= self._cl[1,1,200]
        #143x217 & 217x143
        ell, spec, _ = np.genfromtxt(_get_power_file('gal_planck_143x217'), unpack=True)
        self._cl[1,2,ell.astype(int)] = spec * ell * (ell + 1.0) / (2.0 * np.pi)
        self._cl[1,2] /= self._cl[1,2,200]
        self._cl[2,1,ell.astype(int)] = spec * ell * (ell + 1.0) / (2.0 * np.pi)
        self._cl[2,1] /= self._cl[2,1,200]
        #217x217
        ell, spec, _ = np.genfromtxt(_get_power_file('gal_planck_217'), unpack=True)
        self._cl[2,2,ell.astype(int)] = spec * ell * (ell + 1.0) / (2.0 * np.pi)
        self._cl[2,2] /= self._cl[2,2,200]

        self.set_defaults(**kwargs)

    def eval(self, ell=None, amp=1.):
        """Compute the power spectrum with the given ell and parameters."""
        if np.isscalar(ell):
            ell = np.array(ell)[..., np.newaxis]

        res = np.tile(np.nan, (3,3) + ell.shape)
        res[..., ell <= 2600] = self._cl[..., ell[ell <= 2600]]
        amp = np.array(amp)[..., np.newaxis]

        return amp*res


# CMB template cls?
