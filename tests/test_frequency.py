import fgspectra.frequency as fgf
from fgspectra.model import Model
import numpy as np
from numpy.testing import assert_allclose as aac


def test_bandpass_integration():
    frequency = np.array([[10.0, 20.0, 40.0], [60.0, 90.0, 100.0]])

    transmittance = np.array([[1.0, 2.0, 4.0], [1.0, 3.0, 1.0]])

    class Mock(Model):
        def eval(self, nu=None, par=None):
            if isinstance(nu, list):
                return fgf._bandpass_integration()
            return np.ones_like(nu) * np.array(par)[..., np.newaxis]

    mock = Mock()
    nus = list(zip(frequency, transmittance))

    ress = mock(nus, 1)
    refs = [75.0, 80.0]
    for res, ref in zip(refs, ress):
        aac(res, ref)

    ress = mock(nus, np.array([1.0, 2.0]))
    refs = np.array([[75.0, 80.0], [150.0, 160.0]])
    for res, ref in zip(refs, ress):
        aac(res, ref)
