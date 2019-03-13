# -*- coding: utf-8 -*-
"""
Frequency-dependent foreground components.

This module implements the frequency-dependent component of common foreground
contaminants. This consist only of factorizable components -- i.e. the components
in this module describe f for which the component X has spectrum

.. math:: C_{\ell}^{X} = f(\nu) C^X_{0,\ell}
"""
import numpy as np
from scipy import constants

# With inspiration from BeFoRe and FGBuster by David Alonso and Ben Thorne and
# Davide Poletti and Josquin Errard

class FrequencySpectrum:
    def __init__(self, sed_name=''):
        self.sed = sed_name or something
        return

    def __call__(self, nu, args):
        return self.sed(nu, args)



def things(nu, args, kwargs):
    return nu

# CMB
# blackbody, derivative BB

# Galactic foregrounds
# power law (synch), modified blackbody (dust)
# extensions of these?
# before and fgbuster

# Extragalactic
# mystery
# tile-c

# thermodynamic CMB units - defualt
# other units: Jy/sr or Rayleigh-Jeans, full antenna temp conversion?
