import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import matplotlib.pyplot as plt
import scipy.optimize as op
from scipy.interpolate import InterpolatedUnivariateSpline
import camb
from camb import model, initialpower
import sys
#Get angular power spectrum for galaxy number counts and lensing
from camb.sources import GaussianSourceWindow, SplinedSourceWindow
import glob
import collections
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

#import limber2 as L
#from prepare_PT_tables import growth_factor

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.serif'] = 'cm'

matplotlib.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]

# Load parameters
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import camb
from camb import model, initialpower
import numpy as np

# Code to test limber code for C_{ell} against CAMB's calculations
# CAMB pieces are based off: https://camb.readthedocs.io/en/latest/CAMBdemo.html

# Load fiducial P18 Cosmology
# Results from Planck+BAO in last column of Table 2 in https://arxiv.org/pdf/1807.06209.pdf
ombh2 =  0.02247
omch2 = 0.3097*0.677*0.677 - 0.02247
tau =  0.0925
ln10As = 3.0589
ns = 0.96824
H0 =  67.7
h = H0/100.
sig8 = 0.8277
#cosmomc_theta =  1.04101/100.

# From sec 3.2 in https://arxiv.org/pdf/1807.06205.pdf
Tcmb = 2.7255
Neff = 3.046
YHe = 0.2454 #None # Set from BBN consistency

# Neutrinos, 1 massive and 2 massless
# omnuh2 = minimal mass / 93.14 eV
# Set minimal mass = 0.06 eV
mnu = 0.00
num_massive_neutrinos = 0
hierarchy = 'degenerate'

minkh = 1e-4
maxkh = 1e2
nk = 6000

lmax = 2500

p18_cosmo = FlatLambdaCDM(H0=H0,Om0=(omch2+ombh2)/((H0/100.)**2.),Ob0=ombh2/((H0/100.)**2.),Tcmb0=Tcmb,Neff=Neff,m_nu=[0,0,mnu]*u.eV)

pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, neutrino_hierarchy=hierarchy, num_massive_neutrinos=num_massive_neutrinos, YHe=YHe)
pars.set_dark_energy() #re-set defaults
pars.InitPower.set_params(As=np.exp(ln10As)*10**-10,ns=ns,r=0)
#pars.set_matter_power(redshifts=np.array([0.]), kmax=maxkh)

# Use halofit for the nonlinear power spectrum
#pars.NonLinear = model.NonLinear_both
pars.NonLinear = model.NonLinear_both
pars.set_for_lmax(lmax, lens_potential_accuracy=6)

			 
PK_TT = camb.get_matter_power_interpolator(pars, nonlinear=True, 
			hubble_units=True, k_hunit=True, kmax=maxkh,
			 zmax=10,
			 var1 = 'delta_tot', var2 = 'delta_tot')

#dndz = np.loadtxt('/Users/ALEX/Desktop/simulations/CrowCanyon2/Green/Default_HOD/bsml_dndz.txt')
dndz = np.loadtxt('/Users/ALEX/Berkeley/WISE_cross_CMB/cmb_lss_xcorr/cosmology/green_bsml_dndz_finer_bins_HOD16.txt')

#Set up W(z) window functions, later labelled W1, W2. They are two
# realizations of the Gaussian, one using GaussianSourceWindow
# and one using SplinedSourceWindow.  This is to show that CAMB's
# SplinedSourceWindow is highly accurate (1e-6 level)
#zs = np.arange(0, 0.5, 0.02)
#W = np.exp(-(zs - 0.17) ** 2 / 2 / 0.04 ** 2) / np.sqrt(2 * np.pi) / 0.04
#pars.SourceWindows = [SplinedSourceWindow(bias=1.0, dlog10Ndm=-0.2, z=zs,W=W),
#	GaussianSourceWindow(redshift=0.17,source_type='counts',bias=1.0,sigma=0.04,dlog10Ndm=-0.2)]
# Choose redshifts that are *not* exactly where I've tabulated HF power spectra
#zcs = np.linspace(0.2,2.2,6)
#for zc in zcs:
zs = np.arange(0, 4.0, 0.02)
zc = 0.2
W = zs**2 * np.exp(-(zs - zc) ** 2 / 2 / 0.2 ** 2) / np.sqrt(2 * np.pi) / 0.2
#set Want_CMB to true if you also want CMB spectra or correlations
pars.Want_CMB = True	
test = dndz[:,1]
test[test == 0] = 1e-10
pars.SourceWindows = [SplinedSourceWindow(bias=2.1228, dlog10Ndm=0.4, z=np.ascontiguousarray(dndz[:,0]), W=np.ascontiguousarray(dndz[:,1]/np.sum(dndz[:,1])))]
# Turn off all relativistic effects
pars.SourceTerms.counts_density = True
pars.SourceTerms.counts_redshift = False
pars.SourceTerms.counts_lensing = False
pars.SourceTerms.counts_velocity = False
pars.SourceTerms.counts_radial = False
pars.SourceTerms.counts_timedelay = False
pars.SourceTerms.counts_ISW = False
pars.SourceTerms.counts_potential = False
pars.SourceTerms.counts_evolve = False
pars.SourceTerms.limber_phi_lmin = 10
pars.SourceTerms.limber_windows = True
pars.set_matter_power(nonlinear=True)
results = camb.get_results(pars)
# If I turn on the option "use_raw_cl", CAMB outputs C_{\ell}^{\phi g}

# Get CMB correlations and cross-correlations with source
# These are in dimensionless units
# (I verified by plotting the power spectra and comparing them to the Planck 2018 results.
# Need to multiply by 1e12 Tcmb^2 to get in units of microKelvin ^2
# So for ISW, must multiply by 1e6 Tcmb
cls = results.get_source_cls_dict(raw_cl=True)
ls=  np.arange(2, lmax+1)
camb_clkg = cls['PxW1'][2:lmax+1]*(ls*(ls+1))/2.

camb_clgg = cls['W1xW1'][2:lmax+1]