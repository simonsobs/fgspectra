from fgspectra.frequency import FreqModel
import numpy as np
from numpy.testing import assert_allclose as aac


def test_bandpass_integration():
    frequency = np.array([[10.0, 20.0, 40.0], [60.0, 90.0, 100.0]])

    transmittance = np.array([[1.0, 2.0, 4.0], [1.0, 3.0, 1.0]])

    class Mock(FreqModel):
        def eval(self, nu=None, par=None):
            if isinstance(nu, list):
                return self.eval_bandpass(nu=nu, par=par)
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

    # testing the case with transmittance shape [nchannels, nfreqs, ells]
    transmittance = np.array([[[1.0, 4.0], [2.0, 5.0],[4.0, 6.0]], [[1.0, 4.0], [3.0, 5.0], [1.0, 6.0]]])
    nus = list(zip(frequency, transmittance))

    ress = mock(nus, 1)
    refs = np.array([[ 75.,  80.],
       [155., 190.]])

    for res, ref in zip(refs, ress):
        aac(res, ref)
        
    ress = mock(nus, np.array([1.0, 2.0]))
    refs = np.array([[[ 45.,  60.],
        [180., 240.]],

       [[ 90., 180.],
        [225., 300.]],

       [[180.,  60.],
        [270., 360.]]])

    for res, ref in zip(refs, ress):
        aac(res, ref)
