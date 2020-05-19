import pytest

from fgspectra import cross as fgc
from fgspectra import power as fgp
from fgspectra import frequency as fgf
import numpy as np

def test_ksz():
    assert fgp.kSZ_bat() is not None

def test_ACT_models():
    fgp.defaults = {'ell': np.array([2000])}
    # define the models from fgspectra
    ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
    cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
    radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
    cirrus = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())

    # if there are correlations between components, 
    # have to define them in a joined spectrum
    tSZ_and_CIB = fgc.CorrelatedFactorizedCrossSpectrum(
        fgf.Join(fgf.ThermalSZ(), fgf.CIB()), 
        fgp.SZxCIB_Addison2012())

    # for testing purposes we'll also compute the tSZ and clustered CIB alone
    tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
    cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerLaw())


    par = {
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

    fsz = np.array([par['f0_sz']])
    fsynch = np.array([par['f0_synch']])
    fdust = np.array([par['f0_dust']])

    result = (

            par['a_tSZ'] * tsz(
                {'nu':fsz, 'nu_0': par['nu_0']},
                {'ell_0':par['ell_0']}) ,
            par['a_kSZ'] * ksz(
                {'nu':fsz},
                {'ell_0':par['ell_0']}) ,

            par['a_p'] * cibp(
                {'nu': fdust, 'nu_0':par['nu_0'], 'temp':par['T_d'], 'beta':par['beta_p']},
                {'ell_0':par['ell_0'], 'alpha':2}),

            par['a_c'] * cibc(
                {'nu': fdust, 'nu_0':par['nu_0'], 'temp':par['T_d'], 'beta':par['beta_c']},
                {'ell_0':par['ell_0'], 'alpha':2 - par['n_CIBC']}),

            tSZ_and_CIB(
                {'kwseq': (
                    {'nu':fsz, 'nu_0':par['nu_0']},
                    {'nu': fdust, 'nu_0':par['nu_0'], 'temp':par['T_d'], 'beta':par['beta_c']} 
                    )},
                {'kwseq': ( 
                    {'ell_0':par['ell_0'], 
                     'amp':par['a_tSZ']},
                    {'ell_0':par['ell_0'], 
                     'alpha':2-par['n_CIBC'], 'amp':par['a_c']},
                    {'ell_0':par['ell_0'], 
                     'amp': -par['xi'] * np.sqrt(par['a_tSZ'] * par['a_c'])}
                    )}),

            par['a_s'] * radio(
                {'nu': fsynch, 'nu_0':par['nu_0'], 'beta':-0.5 - 2},
                {'ell_0':par['ell_0'], 'alpha':2}) ,
            par['a_g'] * cirrus(
                {'nu': fdust, 'nu_0':par['nu_0'], 'beta':3.8 - 2},
                {'ell_0':par['ell_0'], 'alpha':-0.7})
    )

    """
    The following array `dunk13` was generated by inserting this code block to
    line 188 of `ACT_equa_likelihood.f90` in the Fortran ACT multifrequency
    likelihood.

    ```fortran
    if(il==2000) then
      write(*,*) (f1*f1)/(f0*f0)*amp_tsz*cl_tsz(il)
      write(*,*) amp_ksz*cl_ksz(il)
      write(*,*) amp_d*cl_p(il)*(f1_dust/feff)**(2.d0*beta_d)*(planckratiod1*fluxtempd1)**2.d0
      write(*,*) amp_c*cl_c(il)*(f1_dust/feff)**(2.d0*beta_c)*(planckratiod1*fluxtempd1)**2.d0
      write(*,*) -2.d0*sqrt(amp_c*amp_tsz)*xi*cl_szcib(il)*((f1_dust**beta_c*f1*planckratiod1*fluxtempd1)/(feff**beta_c*f0))
      write(*,*) amp_s*cl_p(il)*(f1_synch/feff)**(2.d0*alpha_s)*fluxtemps1**2.d0
      write(*,*) amp_ge*cl_cir(il)*(f1_dust**2.d0/feff**2.d0)**beta_g*fluxtempd1**2.d0
    endif
    ```
    """
    dunk13 = np.array( [
        4.46395567509345, # tsz
        1.37903889069153, # ksz
        3.02039291213271, # cibp
        4.36478789541715, # cibc
        -0.611930900267790, # cibxtsz
        1.63099887443075, # sync
        1.15557760337592] ) # cirrus

    # for testing purposes, subtract tsz and cibc alone
    # from the combined tsz,cibc,tszxcibc.
    result = np.array([r[0][0][0] for r in result])
    result[-3] -= result[0]
    result[-3] -= result[3] 
    
    tsz_fg, ksz_fg, cibp_fg, cibc_fg, cibxtxz_fg, sync_fg, cirrus_fg = result
    
    assert ( np.abs(tsz_fg-dunk13[0]) / dunk13[0] < 1e-3 )
    assert ( np.abs(ksz_fg-dunk13[1]) / dunk13[1] < 1e-3 )
    assert ( np.abs(cibp_fg-dunk13[2]) / dunk13[2] < 1e-3 )
    assert ( np.abs(cibc_fg-dunk13[3]) / dunk13[3] < 1e-3 )
    assert ( np.abs(cibxtxz_fg-dunk13[4]) / dunk13[4] < 1e-3 )
    assert ( np.abs(sync_fg-dunk13[5]) / dunk13[5] < 1e-3 )
    assert ( np.abs(cirrus_fg-dunk13[6]) / dunk13[6] < 1e-3 )
