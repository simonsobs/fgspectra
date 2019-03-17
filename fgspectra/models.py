# -*- coding: utf-8 -*-
r"""
Foreground models.

This module draws inspiration from FGBuster (Davide Poletti and Josquin Errard)
and BeFoRe (David Alonso and Ben Thorne).
"""
import fgspectra.frequency, fgspectra.power
import numpy as np

class FGModel:
    def __init__(self):
        self.params = {}
        return
    
    def model(self, nu_1, nu_2, ell, **kwargs):
        """Compute the model at frequency and ell combinations.
        
        Specific models will override this method.
        
        Parameters
        ----------
        nu_1 : float
            first frequency channel in cross-spectrum
        nu_2 : float
            second frequency channel in cross-spectrum
        ell : 1D array
            ells at which we evaluate the model
        **kwargs :
            additional parameters for the model
            
        Returns
        -------
        1D array of floats : model result of same shape as input ells
        """
        raise NotImplementedError
        
    def get_missing(self, input_dict):
        """Copy self.params and ensure no values are None."""
        params = self.params.copy()
        for p in params:
            if (params[p] is None) and (p in input_dict):
                params[p] = input_dict[p]
                
        missing_params = [p for p in params if params[p] is None]
        if len(missing_params) > 0:
            raise Exception(f'Missing parameters: {missing_params}.')
            
        return params
    
    def get_mix(self, freqs, l_max, **kwargs):
        """Compute the mixing matrix up to some ell.
        
        Parameters
        ----------
        freqs : array of floats
            Compute the mixing matrix containing cross-spectra at single
            frequencies, i.e. delta function bandpowers. For example,
            ACT-E has frequencies=[148.0, 220.0].

        l_max : int
            Maximum multipole to compute mixing matrix.

        Returns
        -------
        3D array, first two dimensions correspond to frequency index
        and third dimension is multipole.
        """
        N_freq = len(freqs)
        ells = np.arange(l_max+1)
        mix = np.zeros((N_freq, N_freq, l_max+1))
        for i in range(N_freq):
            for j in range(N_freq):
                mix[i,j,:] = self.model(freqs[i], freqs[j], ells, **kwargs)
                
        return mix
        
    def get_mix_bandpowers(self, bandpower_windows, bandpower_frequencies):
        raise NotImplementedError # gotta put in bandpower integration


class ThermalSZ(FGModel):
    """Model for Thermal Sunyaev-Zel'dovich (Dunkley et al. 2013)."""
    
    def __init__(self, nu_0, ell_0, a_tSZ=None):
        """Initialize the ThermalSZ object with required parameters."""
        
        self.params = { 'nu_0' : nu_0,
                        'ell_0' : ell_0,
                        'a_tSZ' : a_tSZ}
        
        self.fnu = fgspectra.frequency.ThermalSZFreq() # f(nu) template
        self.Dl_template = fgspectra.power.tSZ_150_bat() # D_ell template
        
        return
            
    def model(self, nu_1, nu_2, ell, **kwargs):
        """Compute the Thermal SZ at frequency and ell combinations.
        
        Specific models will override this method.
        
        Parameters
        ----------
        nu_1 : float
            first frequency channel in cross-spectrum
        nu_2 : float
            second frequency channel in cross-spectrum
        ells : 1D array
            ells at which we evaluate the model
        a_tSZ : float
            overall amplitude of the tSZ model
            
        Returns
        -------
        1D array of floats : model result of same shape as input ells
        """
        par = self.get_missing(kwargs)
        print(nu_1, nu_2, par)
        return par['a_tSZ'] * (
            self.fnu(nu_1) * self.fnu(nu_2) / self.fnu(par['nu_0'])**2 *
            self.Dl_template(ell) / self.Dl_template(par['ell_0']) )
            

