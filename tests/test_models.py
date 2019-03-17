import fgspectra.models
import numpy as np

def test_tsz():
    tSZ1 = fgspectra.models.ThermalSZ(nu_0=150.0, ell_0=3000.0)
    assert tSZ1.model( 150.0, 150.0, ell=np.array([3000]), a_tSZ=1.0) == 1.0
