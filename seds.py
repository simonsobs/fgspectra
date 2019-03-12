import numpy as np
from scipy import constants

# With inspiration from BeFoRe and FGBuster by David Alonso and Ben Thorne and 
# Davide Poletti and Josquin Errard

class FrequencyScaling:
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
