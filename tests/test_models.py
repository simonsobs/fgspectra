%matplotlib inline
%load_ext autoreload
%autoreload 2

# %%
from fgspectra import cross as fgc
from fgspectra import power as fgp
from fgspectra import frequency as fgf
import numpy as np

# %%
tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
ksz = fgc.FactorizedCrossSpectrum(fgf.UnitSED(), fgp.kSZ_bat())
cibp = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerLaw())
cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerLaw())
tSZxCIB = fgc.CorrelatedFactorizedCrossSpectrum(
    fgf.tSZxCIB(), fgp.sz_x_cib_template())
radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw_g(), fgp.PowerLaw())
cirrus = fgc.FactorizedCrossSpectrum(fgf.PowerLaw_g(), fgp.PowerLaw())

freqs = np.array([148.0])
ells = np.array([3000])


test_params = {
    'nu_0' : 150.0,
    'ell_0' : 3000,
    'T_CMB' : 2.725,
    'T_d' : 9.7,

    'a_tSZ' : 4.66,
    'a_kSZ' : 1.60,
    'a_p' : 6.87,
    'beta_p' : 2.08,
    'a_c' : 6.10,
    'beta_c' : 2.08,
    'n_CIBC' : 1.20,
    'xi' : 0.09,
    'a_s' :3.50,
    'a_g' :0.88,


    'f0_sz' : 146.9,
    'f0_synch'  :147.6,
    'f0_dust'   :149.7
}

def get_model(par, freqs, ells):
    fsz = np.array([par['f0_sz']])
    fsynch = np.array([par['f0_synch']])
    fdust = np.array([par['f0_dust']])
    print( (-par['xi'] * np.sqrt(par['a_tSZ'] * par['a_c'])) * tSZxCIB(
        [fsz, fdust, par['beta_c'], par['nu_0']], [ells]) )
    return (
        par['a_tSZ'] * tsz(
            [fsz, par['nu_0'], par['T_CMB']], [ells])[0][0][0],
        par['a_kSZ'] * ksz(
            [fsz], [ells])[0][0][0],
        par['a_p'] * cibp(
            [fdust, par['beta_p'], par['T_d'],par['nu_0']],
            [ells, 2, par['ell_0']])[0][0][0] ,
        par['a_c'] * cibc(
            [fdust, par['beta_c'], par['T_d'], par['nu_0']],
            [ells, 2-par['n_CIBC'], par['ell_0']])[0][0][0] ,
        (-par['xi'] * np.sqrt(par['a_tSZ'] * par['a_c'])) * tSZxCIB(
            [fsz, fdust, par['beta_c'], par['nu_0']], [ells])[0][0][0] ,
        par['a_s'] * radio(
            [fsynch, -0.5, par['nu_0'], par['T_CMB']],
            [ells, 2, par['ell_0']])[0][0][0] ,
        par['a_g'] * cirrus(
            [fdust, 3.8, par['nu_0'], par['T_CMB']],
            [ells, -0.7, par['ell_0']])[0][0][0])

# %%
dunk13 = np.array( [5.04810070670028, 1.60000000000000, 6.79588405229860,
                    6.03620247694243, -0.993450058059464, 3.66974746746919, 0.870032016703448] )

get_model(test_params, freqs, ells)
