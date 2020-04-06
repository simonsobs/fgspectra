import fgspectra.frequency as fgf
from fgspectra.model import Model
import numpy as np
from numpy.testing import assert_allclose as aac


def test_bandpass_integration():
    frequency = np.array([[10., 20., 40.],
                          [60., 90., 100.]])

    transmittance = np.array([[1., 2., 4.],
                              [1., 3., 1.]])

    class Mock(Model):
        def eval(self, nu=None, par=None):
            if isinstance(nu, list):
                return fgf._bandpass_integration()
            return np.ones_like(nu) * np.array(par)[..., np.newaxis]

    mock = Mock()
    nus = list(zip(frequency, transmittance))

    ress = mock(nus, 1)
    refs = [75., 80.]
    for res, ref in zip(refs, ress):
        aac(res, ref)

    ress = mock(nus, np.array([1., 2.]))
    refs = np.array([[75., 80.], [150., 160.]])
    for res, ref in zip(refs, ress):
        aac(res, ref)
