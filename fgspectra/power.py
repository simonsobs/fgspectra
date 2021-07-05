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
from .model import Model, _apply


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
        return amp * self._cl[..., np.rint(ell).astype('int')] / \
               self._cl[..., np.rint(ell_0).astype('int'), np.newaxis]

    def diff(self, **kwargs):
        """
        Parameters
        ----------
        ell: float or array
            Multipole
        ell_0: float
            Reference ells
        amp: float or array
            Amplitude,

        Returns
        -------
        cl_diff: dict
            Each key of the dict corresponds to a parameter of the model.
        """
        if 'ell' in kwargs or 'ell_0' in kwargs:
            raise NotImplementedError(
                'Derivatives with respect to ell and ell_0 are not implemented')

        defaults = self.defaults

        if defaults['amp'] is not None:
            return {}

        amp = np.asarray(kwargs['amp'])
        ell = defaults['ell']
        ell_0 = defaults['ell_0']
        res = np.zeros((amp.size, amp.size, ell.size))

        np.einsum('aal->al', res)[:] = self.eval(ell=ell, ell_0=ell_0, amp=1)
        res_amp = res.reshape((amp.size,) + amp.shape + ell.shape)
        return {'amp': res_amp}


class tSZ_150_bat(PowerSpectrumFromFile):
    """PowerSpectrum for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self, **kwargs):
        """Intialize object with parameters."""
        super().__init__(_get_power_file('tsz_150_bat'), **kwargs)


class kSZ_bat(PowerSpectrumFromFile):
    """PowerSpectrum for Kinematic Sunyaev-Zel'dovich (Dunkley et al. 2013)."""

    def __init__(self, **kwargs):
        """Intialize object with parameters."""
        super().__init__(_get_power_file('ksz_bat'), **kwargs)


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

    def diff(self, **kwargs):
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
        cl_diff: dict
            Each key of the dict corresponds to a parameter of the model.
        """
        if 'ell' in kwargs or 'ell_0' in kwargs:
            raise NotImplementedError(
                'Derivatives with respect to ell and ell_0 are not implemented')

        defaults = self.defaults
        res = {}

        alpha = defaults['alpha']
        if alpha is None:
            alpha = kwargs['alpha']

        amp = defaults['amp']
        if amp is None:
            amp = np.asarray(kwargs['amp'])
            res['amp'] = self.eval(alpha=alpha, amp=1.0)[None]
        elif type(amp) in [int,float]:
            amp = np.array([amp])

        if amp.size != 1 or amp.ndim > 1:
            raise NotImplementedError('amp has to be a scalar for now')

        if defaults['alpha'] is None:
            alpha = np.asarray(kwargs['alpha'])
            ell = defaults['ell']
            ell_0 = defaults['ell_0']
            res_alpha = np.zeros((alpha.size, alpha.size, ell.size))

            np.einsum('aal->al', res_alpha)[:] = (
                amp * np.log(ell / ell_0)
                * (ell / ell_0)**(alpha.reshape(-1, 1))
                )
            res['alpha'] = res_alpha.reshape(
                (alpha.size,) + alpha.shape + ell.shape)

        return res


class FreeCls(Model):
    """
    Model with free Cls.
    """

    def eval(self, ell=None, cls=None):
        """ Evaluation of the model

        Parameters
        ----------
        ell : float or array
            Multipoles at which model will be evaluated. If array, the shape
            is ``(ells)``.
        cls : ndarray
            cls of the component. Shape is ``(ells)``
        Returns
        -------
        res : ndarray
            Shape is ``(ells)``
        """
        if type(ell) in (float, int):
            ell = [ell]
        if type(cls) in (float, int):
            cls = [cls]
        try:
            assert len(ell) == len(cls)
        except AssertionError:
            print('Cls must have same size as ells')
        res = cls
        return res

    def diff(self, **kwargs):
        """ Evaluation of the first derivative of the power spectrum.

        Parameters
        ----------
        ell : float or array
            Multipoles at which model will be evaluated. If array, the shape
            is ``(ells)``.
        cls : ndarray
            cls of the component. Shape is ``(ells)``
        Returns
        -------
        cl_diff: dict
            Each key of the dict corresponds to a parameter of the model.
        """
        if 'ell' in kwargs:
            raise NotImplementedError(
                'Derivative with respect to ell does not make sense here')

        defaults = self.defaults
        if defaults['cls'] is not None:
            return {}
        cls = np.asarray(kwargs['cls'])

        # Convention: avoid using eye (wasteful for large number of multipoles)
        # Reverted to old method for now
        #return {'cls': np.ones((cls.size, 1))}
        return {'cls': np.eye(cls.size)}



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
        return {'kwseq': [ps.defaults for ps in self._power_spectra]}

    def _get_repr(self):
        return {type(self).__name__:
                    [ps._get_repr() for ps in self._power_spectra]}


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
        if kwseq:
            spectra = np.array([ps(**kwargs) for ps, kwargs
                                in zip(self._power_spectra, kwseq)])
        else:
            spectra = np.array([ps() for ps in self._power_spectra])
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

    def diff(self, kwseq=None):
        """"Compute the first derivative of the cls."""
        if kwseq is None:
            kwseq = self.defaults['kwseq']
        def diff_in_cls(diff, i, j):
            shape = diff.shape
            res = np.zeros((shape[0], self.n_comp, self.n_comp, shape[-1]))
            res[..., i, j, :] = diff
            res[..., j, i, :] = res[..., i, j, :]
            return res
        diffs = []
        i_corr = 0
        for k_off_diag in range(0, self.n_comp):
            for el_off_diag in range(self.n_comp - k_off_diag):
                i = el_off_diag
                j = el_off_diag + k_off_diag
                diffs.append(_apply(diff_in_cls, self._power_spectra[i_corr].diff(**kwseq[i_corr]), i=i, j=j))
                i_corr += 1
        return {'kwseq': diffs}





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


# CMB template cls?
