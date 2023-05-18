import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from astropy.modeling import models
from astropy import units as u
from astropy.modeling.models import BlackBody
# from astropy.modeling import models
from astropy.modeling.models import Voigt1D

from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from specutils.fitting import fit_continuum
# from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler
import os
from astropy.io import ascii
from astropy.io import fits

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy import interpolate
from scipy.misc import derivative
from scipy.interpolate import UnivariateSpline

from astropy.convolution import convolve, Box1DKernel

#from PyAstronomy import pyasl

from astroquery.ipac.irsa.irsa_dust import IrsaDust
import astropy.coordinates as coord
from dust_extinction.parameter_averages import CCM89

#from rascal.calibrator import Calibrator
#from rascal.util import refine_peaks

#import warnings

plt.rcParams['figure.figsize'] = (12,8)
plt.rcParams['font.size'] = 20
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 17
plt.rcParams['axes.labelsize'] = 30
# plt.rcParams['axes.formatter.useoffset'] = False

model_folders = os.listdir('/Users/crawford/Desktop/models/MODELS_VIS_Courtney-Sept2020/')

all_models = dict()
for folder in model_folders:
    all_models[folder] = dict()
    in_this_folder = os.listdir('/Users/crawford/Desktop/models/MODELS_VIS_Courtney-Sept2020/' + folder)
    for file_name in in_this_folder:
        file_temperature = file_name[:4]
#         print(file_temperature)
        all_models[folder][file_temperature] = folder + '/' + file_name

def which_script(spectra_type):
    if spectra_type == 'AB':
        print('norm_spectra_AB, redfile, bluefile, altredfile, altbluefile')
    elif spectra_type == 'normal':
        print('norm_spectra, redfile, bluefile')
    elif spectra_type == 'AAT':
        print('AAT_spec_norm, one bluefile')
    elif spectra_type == 'one':
        print('norm_spectra_one, one bluefile')
    elif spectra_type == 'ntt':
        print('spec_norm_ntt, one bluefile')
    elif spectra_type == 'ntt_flip':
        print('spec_norm_ntt_flip, one bluefile')
    elif spectra_type == 'txt':
        print('norm_spectra_txt, one textfile')
    elif spectra_type == 'nsv':
        print('NSV11154, one bluefile')
    else:
        print('invalid type, try again.')
        
def blackbody_lam(lam_AA, T):
    from scipy.constants import h,k,c
    lam = 1e-10 * lam_AA # convert to metres
    bb = 2*h*c**2 / (lam_AA**5 * (np.exp(h*c / (lam*k*T)) - 1))
    return bb/np.max(bb)

def voigt_fitter(lam,peak_location,lorentz_amp,lorentz_fwhm,gauss_fwhm):
    v_function = Voigt1D(x_0=peak_location, amplitude_L=lorentz_amp,fwhm_L=lorentz_fwhm,fwhm_G=gauss_fwhm)
    return v_function(lam) #* -1

def norm_spectra(folder,redfile,bluefile,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
    file_path = os.path.join(folder, bluefile)
    specblue = Spectrum1D.read(file_path, format='wcs1d-fits')
    file_path = os.path.join(folder, redfile)
    specred = Spectrum1D.read(file_path, format='wcs1d-fits')
    
    os.chdir(folder)
    specb = fits.open(bluefile)
#     print('Telescope:', specb[0].header['TELESCOP'])
#     print('Instrument:', specb[0].header['INSTRUME'])
#     print('Grating:', specb[0].header['GRATINGR'])
    blue = np.zeros(specb[0].header['NAXIS1'])
    blue[0] = -specb[0].header['CRPIX1']*specb[0].header['CDELT1']+specb[0].header['CRVAL1']
    for i in range(1,specb[0].header['NAXIS1']):
        blue[i] = blue[i-1] + specb[0].header['CDELT1']
#     print(blue)
    blueunits = blue*u.Angstrom
#     print(blueunits)
    
    specr = fits.open(redfile)
    red = np.zeros(specr[0].header['NAXIS1'])
    red[0] = -specr[0].header['CRPIX1']*specr[0].header['CDELT1']+specr[0].header['CRVAL1']
    for i in range(1,specr[0].header['NAXIS1']):
        red[i] = red[i-1] + specr[0].header['CDELT1']
#     print(red)
    redunits = red*u.Angstrom
#     print(redunits)
    os.chdir('..')
    
    if suppress_output==False:
        f,ax = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)
    
    join_point = np.max(blue)
    red_adjust = np.where(red>join_point, red, 0)
    red_trim = np.trim_zeros(red_adjust)
    length = len(red_trim)
#     print(specred.flux[-length:])
    
    lam = np.concatenate((blue, red_trim))
    flux = np.concatenate((specblue.flux,specred.flux[-length:]))
#     ax[0].plot(blue,specblue.flux)
#     ax[0].plot(red,specred.flux)
    if suppress_output==False:
        ax[0].plot(lam,flux)
#     ax[0].set_xlim(5500,6000)
#     plt.show()
    
    spectrum = Spectrum1D(flux=flux*u.Jy, spectral_axis=lam*u.Angstrom)
    g1_fit = fit_continuum(spectrum)
    continuum_fitted = g1_fit(lam*u.Angstrom)
    if suppress_output==False:
        ax[1].plot(lam,flux)
        ax[1].plot(lam,continuum_fitted,linewidth=3)
        plt.show()
    
    if suppress_output==False:
        if toi.isdecimal() == True:
            plt.plot(lam,flux/continuum_fitted,color='k',label='ToI-%s' % toi)
        else:
            plt.plot(lam,flux/continuum_fitted,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    
#     print(lam)
    
    os.chdir("..")
#     print('worked')
    return lam, flux, continuum_fitted


def norm_spectra_AB(folder,redfile1,bluefile1,redfile2,bluefile2,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
    file_path = os.path.join(folder, bluefile1)
    specblue1 = Spectrum1D.read(file_path, format='wcs1d-fits')
    file_path = os.path.join(folder, redfile1)
    specred1 = Spectrum1D.read(file_path, format='wcs1d-fits')
    file_path = os.path.join(folder, bluefile2)
    specblue2 = Spectrum1D.read(file_path, format='wcs1d-fits')
    file_path = os.path.join(folder, redfile2)
    specred2 = Spectrum1D.read(file_path, format='wcs1d-fits')
    
    os.chdir(folder)
    specb1 = fits.open(bluefile1)
    blue1 = np.zeros(specb1[0].header['NAXIS1'])
    blue1[0] = -specb1[0].header['CRPIX1']*specb1[0].header['CDELT1']+specb1[0].header['CRVAL1']
    for i in range(1,specb1[0].header['NAXIS1']):
        blue1[i] = blue1[i-1] + specb1[0].header['CDELT1']
    blue1units = blue1*u.Angstrom
    
    specb2 = fits.open(bluefile2)
    blue2 = np.zeros(specb2[0].header['NAXIS1'])
    blue2[0] = -specb2[0].header['CRPIX1']*specb2[0].header['CDELT1']+specb2[0].header['CRVAL1']
    for i in range(1,specb2[0].header['NAXIS1']):
        blue2[i] = blue2[i-1] + specb2[0].header['CDELT1']
    blue2units = blue2*u.Angstrom
    
    specr1 = fits.open(redfile1)
    red1 = np.zeros(specr1[0].header['NAXIS1'])
    red1[0] = -specr1[0].header['CRPIX1']*specr1[0].header['CDELT1']+specr1[0].header['CRVAL1']
    for i in range(1,specr1[0].header['NAXIS1']):
        red1[i] = red1[i-1] + specr1[0].header['CDELT1']
    red1units = red1*u.Angstrom
    
    specr2 = fits.open(redfile2)
    red2 = np.zeros(specr2[0].header['NAXIS1'])
    red2[0] = -specr2[0].header['CRPIX1']*specr2[0].header['CDELT1']+specr2[0].header['CRVAL1']
    for i in range(1,specr2[0].header['NAXIS1']):
        red2[i] = red2[i-1] + specr2[0].header['CDELT1']
    red2units = red2*u.Angstrom
    os.chdir('..')
    
    if suppress_output==False:
        f,ax = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)
    
#     join_point = np.max(blue)
#     red_adjust = np.where(red>join_point, red, 0)
#     red_trim = np.trim_zeros(red_adjust)
#     length = len(red_trim)
#     print(specred.flux[-length:])
    
    lam = np.concatenate((blue1,blue2,red1,red2))
    flux = np.concatenate((specblue1.flux,specblue2.flux,specred1.flux,specred2.flux))
#     ax[0].plot(blue,specblue.flux)
#     ax[0].plot(red,specred.flux)
    if suppress_output==False:
        ax[0].plot(lam,flux)
#     ax[0].set_xlim(5500,6000)
#     plt.show()
    
    spectrum = Spectrum1D(flux=flux*u.Jy, spectral_axis=lam*u.Angstrom)
    g1_fit = fit_continuum(spectrum)
    continuum_fitted = g1_fit(lam*u.Angstrom)
    if suppress_output==False:
        ax[1].plot(lam,flux)
        ax[1].plot(lam,continuum_fitted,linewidth=3)
        plt.show()
    
    if suppress_output==False:
        if toi.isdecimal() == True:
            plt.plot(lam,flux/continuum_fitted,color='k',label='ToI-%s' % toi)
        else:
            plt.plot(lam,flux/continuum_fitted,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    
#     print(lam)
    
    os.chdir("..")
#     print('worked')
    return lam, flux, continuum_fitted



def norm_spectra_one(folder,redfile,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
    #file_path = os.path.join(folder, bluefile)
    #specblue = Spectrum1D.read(file_path, format='wcs1d-fits')
    file_path = os.path.join(folder, redfile)
    specred = Spectrum1D.read(file_path, format='wcs1d-fits')
    
    os.chdir(folder)
#     specb = fits.open(bluefile)
#     blue = np.zeros(specb[0].header['NAXIS1'])
#     blue[0] = specb[0].header['CRVAL1']
#     for i in range(1,specb[0].header['NAXIS1']):
#         blue[i] = blue[i-1] + specb[0].header['CDELT1']
# #     print(blue)
#     blueunits = blue*u.Angstrom
#     print(blueunits)
    
    specr = fits.open(redfile)
    red = np.zeros(specr[0].header['NAXIS1'])
    red[0] = -specr[0].header['CRPIX1']*specr[0].header['CDELT1']+specr[0].header['CRVAL1']
    for i in range(1,specr[0].header['NAXIS1']):
        red[i] = red[i-1] + specr[0].header['CDELT1']
#     print(red)
    redunits = red*u.Angstrom
#     print(redunits)
    os.chdir('..')
    
    if suppress_output==False:
        f,ax = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)
    
    lam = red
    flux = specred.flux
    
#     lam = np.concatenate((blue, red))
#     flux = np.concatenate((specblue.flux,specred.flux))
#     ax[0].plot(blue,specblue.flux)
    if suppress_output==False:
        ax[0].plot(red,specred.flux)
#     plt.show()
    
#     spectrumb = Spectrum1D(flux=specblue.flux, spectral_axis=blueunits)
#     g1_fit_b = fit_continuum(spectrumb)
#     continuum_fitted_b = g1_fit_b(blueunits)
#     ax[1].plot(blue,specblue.flux)
#     ax[1].plot(blue,continuum_fitted_b,linewidth=3)
    
    spectrumr = Spectrum1D(flux=specred.flux, spectral_axis=redunits)
    g1_fit_r = fit_continuum(spectrumr)
    continuum_fitted_r = g1_fit_r(redunits)
    if suppress_output==False:
        ax[1].plot(red,specred.flux)
        ax[1].plot(red,continuum_fitted_r,linewidth=3)
        plt.show()
    
#     plt.plot(blue,specblue.flux/(continuum_fitted_b*4),color='k',label='ToI-%s' % toi)
    if suppress_output==False:
        if toi.isdecimal() == True:
            plt.plot(lam,specred.flux/continuum_fitted_r,color='k',label='ToI-%s' % toi)
        else:
            plt.plot(lam,specred.flux/continuum_fitted_r,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    
#     print(lam)
    
    os.chdir("..")
#     print('worked')
    return red, specred.flux, continuum_fitted_r


def norm_spectra_txt(folder,txtfile,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
    
    os.chdir(folder)
    spec = np.loadtxt(txtfile)
    lam = spec[:,0]
    flux = spec[:,1]
    os.chdir('..')
    
    if suppress_output==False:
        f,ax = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)

        ax[0].plot(lam,flux)
    
    spectrum = Spectrum1D(flux=flux*u.Jy, spectral_axis=lam*u.Angstrom)
    g1_fit = fit_continuum(spectrum)
    continuum_fitted = g1_fit(lam*u.Angstrom)
    if suppress_output==False:
        ax[1].plot(lam,flux)
        ax[1].plot(lam,continuum_fitted,linewidth=3)
        plt.show()

        if toi.isdecimal() == True:
            plt.plot(lam,flux/continuum_fitted,color='k',label='ToI-%s' % toi)
        else:
            plt.plot(lam,flux/continuum_fitted,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    
    os.chdir("..")
    return lam, flux, continuum_fitted


def norm_spectra_txt_multiple(folder,txtfile,stringlist,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
    
    filelist = stringlist.split(',')
    
    os.chdir(folder)
    spec = np.loadtxt(txtfile)
    lam = spec[:,0]
    flux = spec[:,1]
    
#     plt.plot(lam,flux)
#     plt.show()
    
    flux_i = [[]]
    
    flux_i.append(flux)
    
    for i in range(0,len(filelist)):
        spec1 = np.loadtxt(filelist[i])
        lam1 = spec1[:,0]
        flux1 = spec1[:,1]
        flux_i.append(flux1)
#         plt.plot(lam1,flux1)
#         plt.show()

    flux_i.pop(0)
#     print(flux_i)
    
    median = np.median(flux_i, axis=0)
    print(median)
#         lam = np.concatenate(np.asarray(lam),np.asarray(lam1))
#         flux = np.concatenate(np.asarray(flux),np.asarray(flux1))

    data = np.array([lam, median])
    data = data.T
    #here you transpose your data, so to have it in two columns

    datafile_path = "/Users/crawford/Desktop/Confirmed_RCB/V532Oph/median_combine.txt"
    with open(datafile_path, 'w+') as datafile_id:
    #here you open the ascii file

        np.savetxt(datafile_id, data, fmt=['%f','%e'])
        #here the ascii file is written. 

    os.chdir('..')
    
    if suppress_output==False:
        f,ax = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)

        ax[0].plot(lam,median)
    
    spectrum = Spectrum1D(flux=median*u.Jy, spectral_axis=lam*u.Angstrom)
    g1_fit = fit_continuum(spectrum)
    continuum_fitted = g1_fit(lam*u.Angstrom)
    if suppress_output==False:
        ax[1].plot(lam,median)
        ax[1].plot(lam,continuum_fitted,linewidth=3)
        plt.show()

        if toi.isdecimal() == True:
            plt.plot(lam,median/continuum_fitted,color='k',label='ToI-%s' % toi)
        else:
            plt.plot(lam,median/continuum_fitted,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    
    os.chdir("..")
    return lam, median, continuum_fitted


def AAT_spec_norm(folder,bluefile,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
        
    os.chdir(folder)
    specb = fits.open(bluefile)  
    blue = np.zeros(specb[0].header['NAXIS1'])
    blue[0] = -specb[0].header['CRPIX1']*specb[0].header['CDELT1']+specb[0].header['CRVAL1']
    for i in range(1,specb[0].header['NAXIS1']):
        blue[i] = blue[i-1] + specb[0].header['CDELT1']
    blueunits = blue*u.Angstrom
    fluxblue = specb[0].data

    os.chdir('..')
    
    if suppress_output==False:
        f,ax = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)
    
    lam = blue
    flux = fluxblue
    if suppress_output==False:
        ax[0].plot(blue,fluxblue)
    
    spectrum = Spectrum1D(flux=flux*u.Jy, spectral_axis=lam*u.Angstrom)
    g1_fit = fit_continuum(spectrum)
    continuum_fitted = g1_fit(lam*u.Angstrom)
    if suppress_output==False:
        ax[1].plot(lam,flux)
        ax[1].plot(lam,continuum_fitted,linewidth=3)
        plt.show()
    
    
        plt.plot(lam,flux/continuum_fitted,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()

    
    os.chdir("..")
    return lam, flux, continuum_fitted


def spec_norm_ntt(folder,bluefile,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
        
    os.chdir(folder)
    specb = fits.open(bluefile)  
    blue = np.zeros(specb[0].header['NAXIS1'])
    blue[0] = -specb[0].header['CRPIX1']*specb[0].header['CD1_1']+specb[0].header['CRVAL1']
    for i in range(1,specb[0].header['NAXIS1']):
        blue[i] = blue[i-1] + specb[0].header['CD1_1']
    blueunits = blue*u.Angstrom
    fluxblue = specb[0].data[0,0]
#     print(blueunits)
#     print(fluxblue)
    
    os.chdir('..')
    
    if suppress_output==False:
        f,ax = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)
    
    lam = blue
    flux = fluxblue
    if suppress_output==False:
        ax[0].plot(blue,fluxblue)
    
    spectrum = Spectrum1D(flux=flux*u.Jy, spectral_axis=lam*u.Angstrom)
    g1_fit = fit_continuum(spectrum)
    continuum_fitted = g1_fit(lam*u.Angstrom)
    if suppress_output==False:
        ax[1].plot(lam,flux)
        ax[1].plot(lam,continuum_fitted,linewidth=3)
        plt.show()
    
    
        if toi.isdecimal() == True:
            plt.plot(lam,flux/continuum_fitted,color='k',label='ToI-%s' % toi)
        else:
            plt.plot(lam,flux/continuum_fitted,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    
    
    os.chdir("..")
    return lam, flux, continuum_fitted

def spec_norm_ntt_flip(folder,bluefile,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
        
    os.chdir(folder)
    specb = fits.open(bluefile)  
    blue = np.zeros(specb[0].header['NAXIS1'])
    blue[0] = -specb[0].header['CRPIX1']*specb[0].header['CD1_1']+specb[0].header['CRVAL1']
    for i in range(1,specb[0].header['NAXIS1']):
        blue[i] = blue[i-1] + specb[0].header['CD1_1']
    blueunits = blue*u.Angstrom
    fluxblue = specb[0].data[0,0]
#     print(blueunits)
#     print(fluxblue)
    
    os.chdir('..')
    
    if suppress_output==False:
        f,axis = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)
    
    lam = blue
    flux = np.flip(fluxblue,axis=None)
    if suppress_output==False:
        ax[0].plot(blue,fluxblue)
    
    spectrum = Spectrum1D(flux=flux*u.Jy, spectral_axis=lam*u.Angstrom)
    g1_fit = fit_continuum(spectrum)
    continuum_fitted = g1_fit(lam*u.Angstrom)
    if suppress_output==False:
        ax[1].plot(lam,flux)
        ax[1].plot(lam,continuum_fitted,linewidth=3)
        plt.show()
    
    
        if toi.isdecimal() == True:
            plt.plot(lam,flux/continuum_fitted,color='k',label='ToI-%s' % toi)
        else:
            plt.plot(lam,flux/continuum_fitted,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    
    
    os.chdir("..")
    return lam, flux, continuum_fitted

def NSV11154(folder,bluefile,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
        
    os.chdir(folder)
    specb = fits.open(bluefile)  
    blue = np.zeros(specb[0].header['NAXIS1'])
    blue[0] = -specb[0].header['CRPIX1']*10**specb[0].header['CD1_1']+10**specb[0].header['CRVAL1']
    for i in range(1,specb[0].header['NAXIS1']):
        blue[i] = blue[i-1] + 10**specb[0].header['CD1_1']
#     blueunits = blue*u.Angstrom
    fluxblue = specb[0].data[0]
#     print(blue)
#     print(fluxblue)

    os.chdir('..')
    
    if suppress_output==False:
        f,ax = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)
    
    lam = blue
    flux = fluxblue
    if suppress_output==False:
        ax[0].plot(blue,fluxblue)
    
    spectrum = Spectrum1D(flux=flux*u.Jy, spectral_axis=lam*u.Angstrom)
    g1_fit = fit_continuum(spectrum)
    continuum_fitted = g1_fit(lam*u.Angstrom)
    if suppress_output==False:
        ax[1].plot(lam,flux)
        ax[1].plot(lam,continuum_fitted,linewidth=3)
        plt.show()
    
    
        plt.plot(lam,flux/continuum_fitted,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    
    
    os.chdir("..")
    return lam, flux, continuum_fitted

def convert_radec_to_decimal(rahr,ramin,rasec,decdeg,decmin,decsec,suppress_output=False):
    #1 hr = 15 deg
    hours = rahr + (ramin/60) + (rasec/3600)
    ra_degrees = hours * 15
    if decdeg >= 0:
        dec_degrees = decdeg + (decmin/60) + (decsec/3600)
    else:
        dec_degrees = decdeg - (decmin/60) - (decsec/3600)
    if suppress_output == False:
        print('RA: %d hr %d min %.2f sec -> %.7f degrees' % (rahr,ramin,rasec,ra_degrees))
        print('Dec: %d : %d : %.2f  -> %.7f degrees' % (decdeg,decmin,decsec,dec_degrees))
    elif suppress_output == True:
        pass
    return ra_degrees, dec_degrees
    
def convert_decimal_to_radec(ra_degrees,dec_degrees,suppress_output=False):
    hours = ra_degrees / 15
    rahr = int(hours)
    temp = 60 * (hours - rahr)
    ramin = int(temp)
    rasec = 60 * (temp - ramin)
    if dec_degrees >= 0:
        decdeg = int(dec_degrees)
        temp = 60 * (dec_degrees - decdeg)
        decmin = int(temp)
        decsec = 60 * (temp - decmin)
    else:
        dec_degrees = abs(dec_degrees)
        decdeg = int(dec_degrees)
        temp = 60 * (dec_degrees - decdeg)
        decmin = int(temp)
        decsec = 60 * (temp - decmin)
        decdeg = - decdeg
    if suppress_output == False:
        print('RA: %f degrees -> %d hr %d min %.2f sec' % (ra_degrees,rahr,ramin,rasec))
        print('Dec: %f degrees -> %d : %d : %.2f' % (dec_degrees,decdeg,decmin,decsec))
    elif suppress_output == True:
        pass
    return rahr, ramin, rasec, decdeg, decmin, decsec


def calculate_eq_width_models_catrip(model_folder,lines_list):
    folder = model_folder
    temperatures_list = []
    file_list = os.listdir('/Users/crawford/Desktop/MODELS_VIS_Courtney-Sept2020/' + folder)
    for i in range(0,len(file_list)):
        temperatures_list.append(np.int(file_list[i][:4]))
#     temps[folder] = temperatures_list
    eq_width = dict()
    voigt_params = dict()
    blue_list = []
    red_list = []
    center_list = []
    voigt_red = []
    voigt_blue = []
    voigt_center = []
    
    for temperature in temperatures_list:

        os.chdir('/Users/crawford/Desktop/MODELS_VIS_Courtney-Sept2020/')
        temporary = np.loadtxt(all_models[folder][np.str(temperature)],skiprows=3)
        mod_lam = temporary[:,0]
        mod_flux = temporary[:,1]
        mod_flux_smooth = convolve(mod_flux, Box1DKernel(3))
        os.chdir('/Users/crawford/Desktop/')

        pseudo_cont_left = mod_flux_smooth[np.logical_and(8370 < mod_lam,mod_lam < 8470)]
        lam_pseudo_cont_left = mod_lam[np.logical_and(8370 < mod_lam,mod_lam < 8470)]
        pseudo_cont_right = mod_flux_smooth[np.logical_and(8700 < mod_lam,mod_lam < 8800)]
        lam_pseudo_cont_right = mod_lam[np.logical_and(8700 < mod_lam,mod_lam < 8800)]

        # plt.plot(mod_lam,mod_flux_smooth)
        # plt.plot(lam_pseudo_cont_left,pseudo_cont_left)
        # plt.plot(lam_pseudo_cont_right,pseudo_cont_right)
        # plt.xlim(8000,9000)
        # plt.ylim(-0.5,1.2)
        # plt.show()


        lam_pseudo_cont = np.concatenate((np.asarray(lam_pseudo_cont_left),np.asarray(lam_pseudo_cont_right)))
        pseudo_cont = np.concatenate((np.asarray(pseudo_cont_left),np.asarray(pseudo_cont_right)))

        # plt.plot(mod_lam,mod_flux_smooth)
        # plt.plot(lam_pseudo_cont,pseudo_cont)
        # plt.xlim(8000,9000)
        # plt.ylim(-0.5,1.2)
        # plt.show()


        p = np.polyfit(lam_pseudo_cont,pseudo_cont,1,w=pseudo_cont**2)

        # plt.plot(mod_lam,mod_flux_smooth)
        # plt.plot(lam_pseudo_cont,p[0]*lam_pseudo_cont+p[1])
        # plt.axvline(8498,color='k')
        # plt.axvline(8541,color='k')
        # plt.axvline(8662,color='k')
        # plt.xlim(8000,9000)
        # plt.ylim(-0.5,1.2)
        # plt.show()

        lines = lines_list

        for center in lines:
            
    #     center = 8662
            jumps_left = []
            jumps_right = []
            lam_line_region = mod_lam[np.abs(mod_lam-center)<55]
            flux_line_region = mod_flux_smooth[np.abs(mod_lam-center)<55]
            interp_line_region = interpolate.interp1d(lam_line_region,flux_line_region)
            interp_lam = np.linspace(center-50,center+50,1000)
            interp_flux = interp_line_region(interp_lam)

    #         plt.plot(lam_line_region,flux_line_region)
    #         plt.plot(interp_lam,interp_flux)
    #         plt.show()

            refine_lam_line_region = interp_lam[interp_flux < p[0]*interp_lam+p[1]]
            refine_flux_line_region = interp_flux[interp_flux < p[0]*interp_lam+p[1]]
            for i in range(1,len(refine_lam_line_region)):
                if refine_lam_line_region[i]-refine_lam_line_region[i-1] > 0.2 and refine_lam_line_region[i] < center:
                    jumps_left.append(i)
                elif refine_lam_line_region[i]-refine_lam_line_region[i-1] > 0.2 and refine_lam_line_region[i] > center:
                    jumps_right.append(i)
            if jumps_left == []:
                left_side = refine_lam_line_region[refine_lam_line_region < center]
                flux_left_side = refine_flux_line_region[refine_lam_line_region < center]
                cont_left_side = p[0]*left_side + p[1]
                left_bound = np.argmin(cont_left_side-flux_left_side)
            elif jumps_left != []:
                left_bound = np.max(jumps_left)
            if jumps_right == []:
                right_side = refine_lam_line_region[refine_lam_line_region > center]
                mid_bound = len(refine_lam_line_region)-len(right_side)
                flux_right_side = refine_flux_line_region[refine_lam_line_region > center]
                cont_right_side = p[0]*right_side +p[1]
                right_bound = mid_bound + np.argmin(cont_right_side-flux_right_side)
            elif jumps_right != []:
                right_bound = np.min(jumps_right)
            trim_lam_line_region = refine_lam_line_region[left_bound:right_bound]
            trim_flux_line_region = refine_flux_line_region[left_bound:right_bound]
            cont_line_region = p[0]*trim_lam_line_region + p[1]

            flux_region_to_integrate = (cont_line_region-trim_flux_line_region)/cont_line_region
            
            ## the derivative curve of the spectral line region ##
            
            spl_func_for_deriv = UnivariateSpline(trim_lam_line_region,flux_region_to_integrate,k=4,s=0)
            lam_for_deriv = np.linspace(np.min(trim_lam_line_region),np.max(trim_lam_line_region),1000)
            spl_deriv = spl_func_for_deriv.derivative()
            deriv_roots = spl_deriv.roots()

            deriv_roots_left = deriv_roots[deriv_roots < center-2]
            deriv_roots_right = deriv_roots[deriv_roots > center+2]
            if deriv_roots_left != []:
                deriv_left = np.max(deriv_roots_left)
            else:
                deriv_left = np.min(trim_lam_line_region)
            if deriv_roots_right != []:
                deriv_right = np.min(deriv_roots_right)
            else:
                deriv_right = np.max(trim_lam_line_region)
#             mid_point = (deriv_left+deriv_right)/2
#             diff_limit = deriv_right - mid_point
            
            deriv_lam_left = lam_for_deriv[lam_for_deriv < center-2]
            deriv_lam_right = lam_for_deriv[lam_for_deriv > center+2]
            deriv_array_left = spl_deriv(deriv_lam_left)
            deriv_array_right = spl_deriv(deriv_lam_right)
            left_mask = np.where(deriv_array_left > 0.025,True,False) #true denotes the good parts
            right_mask = np.where(deriv_array_right < -0.025,True,False)
            jumps_left = []
            jumps_right = []
            for i in range(1,len(deriv_lam_left[left_mask])):
                if deriv_lam_left[left_mask][i]-deriv_lam_left[left_mask][i-1] > 0.1:
                    jumps_left.append(deriv_lam_left[left_mask][i])
            for i in range(1,len(deriv_lam_right[right_mask])):
                if deriv_lam_right[right_mask][i]-deriv_lam_right[right_mask][i-1] > 0.1:
                    jumps_right.append(deriv_lam_right[right_mask][i-1])
            if jumps_left == []: ## if array is empty
                left_bound = np.min(deriv_lam_left)
            elif jumps_left != []: ## if array is not empty
                left_bound = np.max(jumps_left)
            if jumps_right == []:
                right_bound = np.max(deriv_lam_right)
            elif jumps_right != []:
                right_bound = np.min(jumps_right)
                
            best_left_bound = max(left_bound,deriv_left)
            best_right_bound = min(right_bound,deriv_right)
            mid_point = (best_left_bound+best_right_bound)/2
            diff_limit = best_right_bound - mid_point
            
#             print(deriv_lam_left[left_mask])
            
#             plt.plot(lam_for_deriv,spl_deriv(lam_for_deriv)) ##### the plotting of the derivative ####
# #             plt.plot(deriv_lam_left[left_bound:right_bound],deriv_array_left[left_mask])
#             plt.axvline(left_bound,color='blue',linewidth=2)
#             plt.axvline(right_bound,color='blue',linewidth=2)
#             plt.axvline(8498,color='k')
#             plt.axvline(8541,color='k')
#             plt.axvline(8662,color='k')
#             plt.axvline(deriv_left,color='red')
#             plt.axvline(deriv_right,color='red')
#             plt.xlim(center-50,center+50)
#             plt.grid(axis='y')
#             plt.show()
                
            new_lam_line_region = trim_lam_line_region[np.abs(trim_lam_line_region-mid_point) < diff_limit]
            new_flux_line_region = trim_flux_line_region[np.abs(trim_lam_line_region-mid_point) < diff_limit]
            cont_line_region = p[0]*new_lam_line_region + p[1]
            
            flux_region_to_integrate = (cont_line_region-new_flux_line_region)/cont_line_region
            
            
            ## add in voigt profile fitting ##
            popt, pcov = curve_fit(voigt_fitter,new_lam_line_region,flux_region_to_integrate,
                                   bounds=((center-1,-np.inf,0,0),(center+1,np.inf,20,20)), maxfev=5000) 
            #had to increase the max number of evaluations. Check here if issues later
            lam_voigt_profile = np.linspace(center-75,center+75,1000)
            voigt_profile = voigt_fitter(lam_voigt_profile,*popt)
            print(popt)
            
            ## to better visualize the fit itself ##
            cont_voigt = p[0]*lam_voigt_profile + p[1]
            voigt_for_plotting = cont_voigt - voigt_profile*cont_voigt

            area = np.round(np.trapz(voigt_profile,lam_voigt_profile)*u.Angstrom,2)
            print('Equivalent Width for', center*u.AA, 'at', temperature*u.K, 'is:', area)

            plt.plot(mod_lam,mod_flux_smooth)
            plt.plot(lam_pseudo_cont,p[0]*lam_pseudo_cont+p[1])
            plt.plot(lam_line_region,flux_line_region)
            plt.plot(refine_lam_line_region,refine_flux_line_region)
            plt.plot(trim_lam_line_region,trim_flux_line_region,linewidth=2)
            plt.plot(new_lam_line_region,new_flux_line_region,linewidth=2)
            plt.plot(lam_voigt_profile,voigt_for_plotting,linewidth=2,color='k')
#             plt.axvline(deriv_left,color='blue')
#             plt.axvline(deriv_right,color='blue')
            plt.axvline(8498,color='k')
            plt.axvline(8541,color='k')
            plt.axvline(8662,color='k')
#             plt.xlim(8200,9000)
            plt.xlim(center-50,center+50)
            plt.ylim(-0.5,1.2)
            plt.show()

            if center == lines[2]:
                red_list.append(area/u.Angstrom)
                voigt_red.append(popt)
            if center == lines[1]:
                center_list.append(area/u.Angstrom)
                voigt_center.append(popt)
            if center == lines[0]:
                blue_list.append(area/u.Angstrom)
                voigt_blue.append(popt)
        
        
    eq_width[lines[0]] = blue_list
    eq_width[lines[1]] = center_list
    eq_width[lines[2]] = red_list

    voigt_params[lines[0]] = voigt_blue
    voigt_params[lines[1]] = voigt_center
    voigt_params[lines[2]] = voigt_red
    
    return eq_width, voigt_params, temperatures_list

def norm_spectra_TNG(folder,bluefile,redfile,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
        
    os.chdir(folder)
    specb = fits.open(bluefile)  
    blue = np.zeros(specb[0].header['NAXIS1'])
    blue[0] = -specb[0].header['CRPIX1']*specb[0].header['CDELT1']+specb[0].header['CRVAL1']
    for i in range(1,specb[0].header['NAXIS1']):
        blue[i] = blue[i-1] + specb[0].header['CDELT1']
    blueunits = blue*u.Angstrom
    fluxblue = specb[0].data
#     print(len(fluxblue),len(blue))
    
    specr = fits.open(redfile)  
    red = np.zeros(specr[0].header['NAXIS1'])
    red[0] = -specr[0].header['CRPIX1']*specr[0].header['CDELT1']+specr[0].header['CRVAL1']
    for i in range(1,specr[0].header['NAXIS1']):
        red[i] = red[i-1] + specr[0].header['CDELT1']
    redunits = red*u.Angstrom
    fluxred = specr[0].data
#     print(len(fluxred),len(red))
    
    os.chdir('..')
    
    if suppress_output==False:
        f,ax = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)
        
    join_point = np.max(blue)
    red_adjust = np.where(red>join_point, red, 0)
    red_trim = np.trim_zeros(red_adjust)
    length = len(red_trim)
#     print(join_point)
#     print(specred.flux[-length:])

    overlap = np.trim_zeros(np.where(red<blue[-1],red,0))
    overlapblue = np.trim_zeros(np.where(blue>red[0],blue,0))

    scale_factor = np.average(fluxred[:len(overlap)])/np.average(fluxblue[-len(overlapblue):])
    lam = np.concatenate((blue, red[len(overlap):]))
    flux = np.concatenate((fluxblue*scale_factor,fluxred[len(overlap):]))

#     print(len(lam),len(flux))
#     ax[0].plot(blue,specblue.flux)
#     ax[0].plot(red,specred.flux)
#     print(lam)
#     print(flux)
    if suppress_output==False:
        ax[0].plot(lam,flux)
#     ax[0].set_xlim(5500,6000)
#     plt.show()
    
#     lam = blue
#     flux = fluxblue
#     if suppress_output==False:
#         ax[0].plot(blue,fluxblue)
    
    spectrum = Spectrum1D(flux=flux*u.Jy, spectral_axis=lam*u.Angstrom)
    g1_fit = fit_continuum(spectrum)
    continuum_fitted = g1_fit(lam*u.Angstrom)
    if suppress_output==False:
        ax[1].plot(lam,flux)
        ax[1].plot(lam,continuum_fitted,linewidth=3)
        plt.show()
    
    
        if toi.isdecimal() == True:
            plt.plot(lam,flux/continuum_fitted,color='k',label='ToI-%s' % toi)
        else:
            plt.plot(lam,flux/continuum_fitted,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    
    
    os.chdir("..")
    return lam, flux, continuum_fitted

def norm_spectra_TNG(folder,bluefile,redfile,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
        
    os.chdir(folder)
    specb = fits.open(bluefile)  
    blue = np.zeros(specb[0].header['NAXIS1'])
    blue[0] = -specb[0].header['CRPIX1']*specb[0].header['CDELT1']+specb[0].header['CRVAL1']
    for i in range(1,specb[0].header['NAXIS1']):
        blue[i] = blue[i-1] + specb[0].header['CDELT1']
    blueunits = blue*u.Angstrom
    fluxblue = specb[0].data
#     print(len(fluxblue),len(blue))
    
    specr = fits.open(redfile)  
    red = np.zeros(specr[0].header['NAXIS1'])
    red[0] = -specr[0].header['CRPIX1']*specr[0].header['CDELT1']+specr[0].header['CRVAL1']
    for i in range(1,specr[0].header['NAXIS1']):
        red[i] = red[i-1] + specr[0].header['CDELT1']
    redunits = red*u.Angstrom
    fluxred = specr[0].data
#     print(len(fluxred),len(red))
    
    os.chdir('..')
    
    if suppress_output==False:
        f,ax = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)
        
    join_point = np.max(blue)
    red_adjust = np.where(red>join_point, red, 0)
    red_trim = np.trim_zeros(red_adjust)
    length = len(red_trim)
#     print(join_point)
#     print(specred.flux[-length:])

    overlap = np.trim_zeros(np.where(red<blue[-1],red,0))
    overlapblue = np.trim_zeros(np.where(blue>red[0],blue,0))

    scale_factor = np.average(fluxred[:len(overlap)])/np.average(fluxblue[-len(overlapblue):])
    lam = np.concatenate((blue, red[len(overlap):]))
    flux = np.concatenate((fluxblue*scale_factor,fluxred[len(overlap):]))

    
#     print(len(lam),len(flux))
#     ax[0].plot(blue,specblue.flux)
#     ax[0].plot(red,specred.flux)
#     print(lam)
#     print(flux)
    if suppress_output==False:
        ax[0].plot(lam,flux)
#     ax[0].set_xlim(5500,6000)
#     plt.show()
    
#     lam = blue
#     flux = fluxblue
#     if suppress_output==False:
#         ax[0].plot(blue,fluxblue)
    
    spectrum = Spectrum1D(flux=flux*u.Jy, spectral_axis=lam*u.Angstrom)
    g1_fit = fit_continuum(spectrum)
    continuum_fitted = g1_fit(lam*u.Angstrom)
    if suppress_output==False:
        ax[1].plot(lam,flux)
        ax[1].plot(lam,continuum_fitted,linewidth=3)
        plt.show()
    
    
        if toi.isdecimal() == True:
            plt.plot(lam,flux/continuum_fitted,color='k',label='ToI-%s' % toi)
        else:
            plt.plot(lam,flux/continuum_fitted,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    
    
    os.chdir("..")
    return lam, flux, continuum_fitted


def norm_spectra_picky(folder,bluefile,redfile,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
        
    os.chdir(folder)
    specb = fits.open(bluefile)  
    blue = np.zeros(specb[0].header['NAXIS1'])
    blue[0] = -specb[0].header['CRPIX1']*specb[0].header['CDELT1']+specb[0].header['CRVAL1']
    for i in range(1,specb[0].header['NAXIS1']):
        blue[i] = blue[i-1] + specb[0].header['CDELT1']
    blueunits = blue*u.Angstrom
    fluxblue = specb[0].data
#     print(len(fluxblue),len(blue))
    
    specr = fits.open(redfile)  
    red = np.zeros(specr[0].header['NAXIS1'])
    red[0] = -specr[0].header['CRPIX1']*specr[0].header['CDELT1']+specr[0].header['CRVAL1']
    for i in range(1,specr[0].header['NAXIS1']):
        red[i] = red[i-1] + specr[0].header['CDELT1']
    redunits = red*u.Angstrom
    fluxred = specr[0].data
#     print(len(fluxred),len(red))
    
    os.chdir('..')
    
    if suppress_output==False:
        f,ax = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)

    join_point = np.max(blue)
    red_adjust = np.where(red>join_point, red, 0)
    red_trim = np.trim_zeros(red_adjust)
    length = len(red_trim)
#     print(join_point)
#     print(specred.flux[-length:])

    overlap = np.trim_zeros(np.where(red<blue[-1],red,0))
    overlapblue = np.trim_zeros(np.where(blue>red[0],blue,0))

    scale_factor = np.average(fluxred[:len(overlap)])/np.average(fluxblue[-len(overlapblue):])
    if toi == 'A182':
        scale_factor = np.average(fluxred[350:len(overlap)])/np.average(fluxblue[-len(overlapblue):])
    lam = np.concatenate((blue[:-int(len(overlapblue)/2)], red[int(len(overlap)/2):]))
    flux = np.concatenate((fluxblue[:-int(len(overlapblue)/2)]*scale_factor,fluxred[int(len(overlap)/2):]))

#     print(len(lam),len(flux))
#     ax[0].plot(blue,specblue.flux)
#     ax[0].plot(red,specred.flux)
    if suppress_output==False:
        ax[0].plot(lam,flux)
#     ax[0].set_xlim(5500,6000)
#     plt.show()
    
#     lam = blue
#     flux = fluxblue
#     if suppress_output==False:
#         ax[0].plot(blue,fluxblue)
    
    spectrum = Spectrum1D(flux=flux*u.Jy, spectral_axis=lam*u.Angstrom)
    g1_fit = fit_continuum(spectrum)
    continuum_fitted = g1_fit(lam*u.Angstrom)
    if suppress_output==False:
        ax[1].plot(lam,flux)
        ax[1].plot(lam,continuum_fitted,linewidth=3)
        plt.show()
    
    
        if toi.isdecimal() == True:
            plt.plot(lam,flux/continuum_fitted,color='k',label='ToI-%s' % toi)
        else:
            plt.plot(lam,flux/continuum_fitted,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    
    
    os.chdir("..")
    return lam, flux, continuum_fitted

def norm_spectra_picky_one(folder,bluefile,toi,suppress_output=True):
    os.chdir("./Confirmed_RCB")
        
    os.chdir(folder)
    specb = fits.open(bluefile)  
    blue = np.zeros(specb[0].header['NAXIS1'])
    blue[0] = -specb[0].header['CRPIX1']*specb[0].header['CDELT1']+specb[0].header['CRVAL1']
    for i in range(1,specb[0].header['NAXIS1']):
        blue[i] = blue[i-1] + specb[0].header['CDELT1']
    blueunits = blue*u.Angstrom
    fluxblue = specb[0].data
    
    os.chdir('..')
    
    lam = blue
    flux = fluxblue
    
    if suppress_output==False:
        f,ax = plt.subplots(2, 1, figsize=[12,8], constrained_layout=True)
        ax[0].plot(lam,flux)
    
    spectrum = Spectrum1D(flux=flux*u.Jy, spectral_axis=lam*u.Angstrom)
    g1_fit = fit_continuum(spectrum)
    continuum_fitted = g1_fit(lam*u.Angstrom)
    if suppress_output==False:
        ax[1].plot(lam,flux)
        ax[1].plot(lam,continuum_fitted,linewidth=3)
        plt.show()
    
    
        if toi.isdecimal() == True:
            plt.plot(lam,flux/continuum_fitted,color='k',label='ToI-%s' % toi)
        else:
            plt.plot(lam,flux/continuum_fitted,color='k',label='%s' % toi)
        plt.ylim(-10,10)
        plt.legend()
        plt.show()
    
    
    os.chdir("..")
    return lam, flux, continuum_fitted

# shift all the stars to the rest frame
# first calculate and store all the redshift values, and then apply those to the wavelength cals
def find_redshift_solved(wavelength,flux,temperature,star_name,suppress_output=True,region='Ca trip'):
    if suppress_output == False:
        print('Calculating Redshift for', star_name)
    lam = wavelength
    flux = flux
    
    os.chdir('/Users/crawford/Desktop/models/MODELS_VIS_Courtney-Sept2020')

    temporary = np.loadtxt(all_models['O1618-1_C1213-10_N70'][np.str(temperature)],skiprows=3)
    mod_lam = temporary[:,0]
    mod_flux = temporary[:,1]
    mod_flux_smooth = convolve(mod_flux, Box1DKernel(3))

    os.chdir('/Users/crawford/Desktop/')

    if region == 'Ca trip':
        cat_region_m = np.where(np.abs(mod_lam-8600)<=250,True,False)
        cat_region_d = np.where(np.abs(lam-8600)<=250,True,False)
    
        if lam[cat_region_d] != [] and lam[cat_region_d][-1] > 8800:

        # Carry out the cross-correlation.
        # The RV-range is -30 - +30 km/s in steps of 0.6 km/s.
        # The first and last 20 points of the data are skipped.
            rv, cc = pyasl.crosscorrRV(lam[cat_region_d], flux[cat_region_d]/np.max(flux[cat_region_d]),mod_lam[cat_region_m], mod_flux_smooth[cat_region_m], -450., 450., 0.1, skipedge=20)

            # Find the index of maximum cross-correlation function
            maxind = np.argmax(cc)
            if suppress_output == False:
                print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")
                if rv[maxind] > 0.0:
                    print("  A red-shift with respect to the template")
                else:
                    print("  A blue-shift with respect to the template")

            rad_vel = rv[maxind]

            if suppress_output == False:
                plt.plot(rv, cc, 'b-')
                plt.plot(rv[maxind], cc[maxind], 'ro')
                plt.show()

            rest_lam = lam - ((rv[maxind]/3e5)*lam)

            if suppress_output == False:
                # Plot template and data
                plt.title("%s Redshift Applied: Template (blue) and data (red)" % (star_name))
                plt.plot(mod_lam[cat_region_m], mod_flux_smooth[cat_region_m], 'b')
                plt.plot(rest_lam[cat_region_d], flux[cat_region_d]/np.max(flux[cat_region_d]), 'r')
                plt.axvline(8498,color='gray')
                plt.axvline(8542,color='gray')
                plt.axvline(8662,color='gray')
                plt.xlim(8350,8800)
                plt.show()
        else:
            if suppress_output == False:
                print('No Calcium Triplet Region')
            rad_vel = 0
            rest_lam = [0,0]
            
    elif region == 'C2 bands':
        cat_region_m = np.where(np.abs(mod_lam-5000)<=500,True,False)
        cat_region_d = np.where(np.abs(lam-5000)<=500,True,False)
    
        if lam[cat_region_d] != [] and lam[cat_region_d][-1] > 5400 and lam[cat_region_d][0] > 4600:

            rv, cc = pyasl.crosscorrRV(lam[cat_region_d], flux[cat_region_d]/np.max(flux[cat_region_d]),mod_lam[cat_region_m], mod_flux_smooth[cat_region_m], -450., 450., 0.1, skipedge=20)
            
            # Find the index of maximum cross-correlation function
            maxind = np.argmax(cc)
            if suppress_output == False:
                print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")
                if rv[maxind] > 0.0:
                    print("  A red-shift with respect to the template")
                else:
                    print("  A blue-shift with respect to the template")

            rad_vel = rv[maxind]

            if suppress_output == False:
                plt.plot(rv, cc, 'b-')
                plt.plot(rv[maxind], cc[maxind], 'ro')
                plt.show()

            rest_lam = lam - ((rv[maxind]/3e5)*lam)

            if suppress_output == False:
                # Plot template and data
                plt.title("%s Redshift Applied: Template (blue) and data (red)" % (star_name))
                plt.plot(mod_lam[cat_region_m], mod_flux_smooth[cat_region_m], 'b')
                plt.plot(rest_lam[cat_region_d], flux[cat_region_d]/np.max(flux[cat_region_d]), 'r')
                #plt.axvline(8498,color='gray')
                #plt.axvline(8542,color='gray')
                #plt.axvline(8662,color='gray')
                plt.xlim(mod_lam[cat_region_m][0],mod_lam[cat_region_m][-1])
                plt.show()
        else:
            if suppress_output == False:
                print('No C2 band 5000-6000 Region')
            rad_vel = 0
            rest_lam = [0,0]
        
    return star_name, rad_vel, rest_lam



def initialize_RCB_spectra():
    os.chdir('/Users/crawford/Desktop')
    master = pd.read_csv('RCB_master_list.csv')
    model_folders = os.listdir('/Users/crawford/Desktop/models/MODELS_VIS_Courtney-Sept2020/')

    all_models = dict()
    for folder in model_folders:
        all_models[folder] = dict()
        in_this_folder = os.listdir('/Users/crawford/Desktop/models/MODELS_VIS_Courtney-Sept2020/' + folder)
        for file_name in in_this_folder:
            file_temperature = file_name[:4]
        #         print(file_temperature)
            all_models[folder][file_temperature] = folder + '/' + file_name
            
    master_w_spec = master[master['Spectra_Type'].notnull()]
    master_w_spec.reset_index(drop=True,inplace=True)

    master_store_lam = [[]]
    master_store_flux = [[]]
    master_store_cont = [[]]
    master_store_type = []
    for i in range(0,len(master_w_spec)):
        if master_w_spec['Spectra_Type'][i]=='AB':
            a,b,c = norm_spectra_AB(master_w_spec['Folder'][i],master_w_spec['Redfile'][i],
                                         master_w_spec['Bluefile'][i],master_w_spec['Alt_Redfile'][i],
                                         master_w_spec['Alt_Bluefile'][i],master_w_spec['Name'][i])
        elif master_w_spec['Spectra_Type'][i]=='normal':
            a,b,c = norm_spectra(master_w_spec['Folder'][i],master_w_spec['Redfile'][i],
                                      master_w_spec['Bluefile'][i],master_w_spec['Name'][i])
        elif master_w_spec['Spectra_Type'][i]=='TNG':
            a,b,c = norm_spectra_TNG(master_w_spec['Folder'][i],master_w_spec['Redfile'][i],
                                     master_w_spec['Bluefile'][i],master_w_spec['Name'][i])
        elif master_w_spec['Spectra_Type'][i]=='AAT':
            temp = master_w_spec.iloc[i]
            a,b,c = AAT_spec_norm(temp['Folder'],temp['Bluefile'],temp['Name'])
        if master_w_spec['Spectra_Type'][i]=='picky':
            a,b,c = norm_spectra_picky(master_w_spec['Folder'][i],master_w_spec['Bluefile'][i],
                                      master_w_spec['Redfile'][i],master_w_spec['Name'][i])
        elif master_w_spec['Spectra_Type'][i]=='one':
            a,b,c = norm_spectra_one(master_w_spec['Folder'][i],master_w_spec['Bluefile'][i],
                                          master_w_spec['Name'][i])
        elif master_w_spec['Spectra_Type'][i]=='pickyone':
            a,b,c = norm_spectra_picky_one(master_w_spec['Folder'][i],master_w_spec['Bluefile'][i],
                                          master_w_spec['Name'][i])
        elif master_w_spec['Spectra_Type'][i]=='ntt':
            temp = master_w_spec.iloc[i]
            a,b,c = spec_norm_ntt(temp['Folder'],temp['Bluefile'],temp['Name'])
        elif master_w_spec['Spectra_Type'][i]=='ntt_flip':
            temp = master_w_spec.iloc[i]
            a,b,c = spec_norm_ntt_flip(temp['Folder'],temp['Bluefile'],temp['Name'])
        elif master_w_spec['Spectra_Type'][i]=='txt':
            a,b,c = norm_spectra_txt(master_w_spec['Folder'][i],master_w_spec['Textfile'][i],
                                          master_w_spec['Name'][i])
        elif master_w_spec['Spectra_Type'][i]=='nsv':
            a,b,c = NSV11154(master_w_spec['Folder'][i],master_w_spec['Bluefile'][i],master_w_spec['Name'][i])
        master_store_lam.append(a)
        master_store_flux.append(b)
        master_store_cont.append(c)
        master_store_type.append(master_w_spec.iloc[i]['Type'])
    master_store_lam.pop(0)
    master_store_flux.pop(0)
    master_store_cont.pop(0)
    # master_store_type.pop(0)
    
    return master_w_spec, master_store_lam, master_store_flux, master_store_cont, master_store_type, all_models

def redshift_RCB_spectra(master_w_spec, master_store_lam, master_store_flux, master_store_type,region='Ca trip',suppress_output=True):
    store_rv = []
    store_names = []
    store_types = []
    store_ra = []
    store_dec = []
    store_rest_wavelength = [[]]
    store_rest_flux = [[]]
    if suppress_output == True:
        output = True
    elif suppress_output == False:
        output = False
    for i in range(0,len(master_w_spec)-6):
        ## skip the stars that are in decline or just wrong
        #if master_w_spec['Name'][i]=='ASAS-RCB-21':
         #   rest_wavelength = [0,0]
        #elif master_w_spec['Name'][i]=='ASAS-RCB-19':
            #rest_wavelength = [0,0]
        if master_w_spec['Name'][i]=='ASAS-RCB-13':
            rest_wavelength = [0,0]
        elif master_w_spec['Name'][i]=='EROS2-CG-RCB-12':
            rest_wavelength = [0,0]
        elif master_w_spec['Name'][i]=='AO Her':
            #rest_wavelength = master_store_lam[i]
            name = master_w_spec['Name'][i]
            radial_velocity = -680
            rest_wavelength = master_store_lam[i] - ((radial_velocity/3e5)*master_store_lam[i])
            star_type = 'Cool'
        #elif master_w_spec['Name'][i]=='HdCcand-C1004':
            #name = master_w_spec['Name'][i]
            #radial_velocity = 291
            #rest_wavelength = master_store_lam[i] - ((radial_velocity/3e5)*master_store_lam[i])
            #star_type = 'Warm'
        #elif master_w_spec['Name'][i]=='HdCcand-B564':
            #name = master_w_spec['Name'][i]
            #radial_velocity = 4
            #rest_wavelength = master_store_lam[i] - ((radial_velocity/3e5)*master_store_lam[i])
            #star_type = 'Warm'
        #elif master_w_spec['Name'][i]=='HdCcand-A980':
            #name = master_w_spec['Name'][i]
            #radial_velocity = 100
            #rest_wavelength = master_store_lam[i] - ((radial_velocity/3e5)*master_store_lam[i])
            #star_type = 'Warm'
        elif master_w_spec['Name'][i]=='MSX-SMC-014':
            rest_wavelength = [0,0]
        else:
            if master_store_type[i] != 'Hot':
                if master_store_type[i] == 'Cool':
                    temperature = 4000
                    star_type = 'Cool'
                elif master_store_type[i] == 'Warm/Cool':
                    temperature = 5500
                    star_type = 'Warm/Cool'
                elif master_store_type[i] == 'Warm':
                    temperature = 7000
                    star_type = 'Warm'
                if region == 'Ca trip':
                    name, radial_velocity, rest_wavelength = find_redshift_solved(master_store_lam[i],master_store_flux[i],temperature,master_w_spec['Name'][i],suppress_output=output)
                elif region == 'C2 bands':
                    name, radial_velocity, rest_wavelength = find_redshift_solved(master_store_lam[i],master_store_flux[i],temperature,master_w_spec['Name'][i],region='C2 bands',suppress_output=output)
            else:
                rest_wavelength = [0,0]
        if rest_wavelength != [0,0]:
            store_rv.append(radial_velocity)
            store_names.append(name)
            store_types.append(star_type)
            store_rest_wavelength.append(rest_wavelength)
            store_rest_flux.append(master_store_flux[i])
            store_ra.append(master_w_spec['RA'][i])
            store_dec.append(master_w_spec['Dec'][i])

    store_rest_wavelength.pop(0)
    store_rest_flux.pop(0)
    
    return store_rv, store_names, store_types, store_rest_wavelength, store_rest_flux, store_ra, store_dec

def read_RCB_extinction_values(master_w_spec,store_names):
    store_Av_values = []
    for i in range(0,len(master_w_spec)):
        if master_w_spec['Name'][i] in store_names:
            store_Av_values.append(master_w_spec['Av'][i])

    return store_Av_values

def initialize_RCB_extinction_values(store_names,store_ra,store_dec):
    green_Av_values = {'J175107.12-242357.3':7.10,
                  'J184158.40-054819.2':4.76,
                  'ASAS-RCB-9':4.00, #not actually from Green, just an estimate based on colors
                  'EROS2-CG-RCB-7':4.00, #check this estimate to see if it could be better
                  'EROS2-CG-RCB-9':4.91,
                       'J172553.80-312421.1':6.84,
                       'J054221.91-690259.3':0.2}
    adjusted_Av = {'J054221.91-690259.3':3.5,'EROS2-SMC-RCB-3':2.3,
                   'J175749.76-075314.9':4.3,'ASAS-RCB-14':4.0,
                   'ASAS-RCB-18':4.5,'ASAS-RCB-19':4.0,'ASAS-RCB-20':3.0,
                   'J132354.47-673720.8':3.0,'V517 Oph':5.0,
                   'J182723.38-200830.1':5.5,'J184246.26-125414.7':5.5,
                   'J182334.24-282957.1':4.0,'J173202.75-432906.1':4.5,
                   'AO Her':2.5}
    ext = CCM89(Rv=3.1)

    store_Av_values = []
    for i in range(0,len(store_names)):
        if store_names[i] in list(green_Av_values.keys()):
            store_Av_values.append(green_Av_values[store_names[i]])
        #elif store_names[i] in list(adjusted_Av.keys()):
        #    store_Av_values.append(adjusted_Av[store_names[i]])
        else:
    #         print(store_names[i])
            ra = store_ra[i]
            dec = store_dec[i]
    #         print(ra,dec)
            rahr = ra[:2]
            ram = ra[3:5]
            ras = ra[6:]
            if dec[0] == '-':
                decdeg = dec[:3]
                decm = dec[4:6]
                decs = dec[7:]
            else:
                decdeg = dec[:2]
                decm = dec[3:5]
                decs = dec[6:]
            rastring = str(rahr)+'h' + str(ram)+'m' + str(ras)+'s'
            decstring = str(decdeg)+'d' + str(decm)+'m' + str(decs)+'s'
            coordstring = rastring + ' ' + decstring
            coo = coord.SkyCoord(coordstring)
            table = IrsaDust.get_extinction_table(coo)
            store_Av_values.append(table['A_SandF'][2])
            
    return store_Av_values

cutoff_dict = {'PV Tel':3600,
               'AO Her': 3600,'ASAS-RCB-1':3600,'ASAS-RCB-3':3600,
               'ASAS-RCB-7':3600,'ASAS-RCB-10':3600,
               'ASAS-RCB-11': 3600, 'ASAS-RCB-12': 3600, 'ASAS-RCB-14': 3600,
               'ASAS-RCB-15': 4000, 'ASAS-RCB-16': 3600, 'ASAS-RCB-17': 4100,
               'ASAS-RCB-18': 4400, 'ASAS-RCB-19': 4250, 'ASAS-RCB-2': 3600,
               'ASAS-RCB-20': 4500, 'ASAS-RCB-4': 3600, 'ASAS-RCB-5': 3600,
               'ASAS-RCB-6': 3700, 'ASAS-RCB-8': 3600, 'ASAS-RCB-9': 4000,
               'ASAS-RCB-21': 3600, 'EROS2-CG-RCB-1': 5000,
               'EROS2-CG-RCB-11': 5500, 'EROS2-CG-RCB-13': 4000,
               'EROS2-CG-RCB-14': 5500, 'EROS2-CG-RCB-3': 4250,
               'EROS2-CG-RCB-4': 5500, 'EROS2-CG-RCB-5': 5500,
               'EROS2-CG-RCB-8': 5750,'EROS2-CG-RCB-10': 5750,
               'EROS2-CG-RCB-6': 4800, 'EROS2-CG-RCB-7': 5500,
               'EROS2-CG-RCB-9': 5500, 'ES Aql': 3600, 'FH Sct': 3600,
               'GU Sgr': 3600, 'HE 1015-2050': 3800, 'IRAS1813.5-2419': 3600,
               'J110008.77-600303.6': 5000, 'J132354.47-673720.8': 3600,
               'J150104.50-563325.1': 5000, 'J160205.48-552741.6': 4250,
               'J161156.23-575527.1': 3600, 'J163450.35-380218.5': 6000,
               'J164704.67-470817.8': 5500, 'J170343.87-385126.6': 5800,
               'J171815.36-341339.9': 5800, 'J171908.50-435044.6': 6000,
               'J172447.52-290418.6': 3700, 'J172553.80-312421.1': 5500,
               'J172951.80-101715.9': 3600, 'J173202.75-432906.1': 3800,
               'J173553.02-364104.3': 4750, 'J173819.81-203632.1': 3800,
               'J174111.80-281955.3': 5000, 'J174119.57-250621.2': 3700,
               'J174138.87-161546.4': 3600, 'J174257.19-362052.1': 5000,
               'J174328.50-375029.0': 3700, 'J174645.90-250314.1': 4500,
               'J174851.29-330617.0': 3800, 'J175031.70-233945.7': 4500,
               'J175107.12-242357.3': 4500, 'J175521.75-281131.2': 5000,
               'J175558.51-164744.3': 3600, 'J175749.76-075314.9': 4500,
               'J175749.98-182522.8': 5000, 'J180550.49-151301.7': 4800,
               'J181252.50-233304.4': 4200, 'J181538.25-203845.7': 4750,
               'J182235.25-033213.2': 3600,
               'J182334.24-282957.1': 4000, 'J182723.38-200830.1': 4000,
               'J182943.83-190246.2': 3800, 'J183649.54-113420.7': 4000,
               'J184158.40-054819.2': 4000, 'J184246.26-125414.7': 4250,
               'J185525.52-025145.7': 4250, 'J194218.38-203247.5': 3600,
               'MACHO 135.27132.51': 3700, 'MACHO 301.45783.9': 3700,
               'MACHO 308.38099.66': 4000, 'MACHO 401.48170.2237': 4200,
               'NSV11154': 4800, 'OGLE-GC-RCB-1': 3800, 'R CrB': 4800,
               'RS Tel': 3600, 'RT Nor': 3600, 'RY Sgr': 3600, 'RZ Nor': 3600,
               'S Aps': 3600, 'SV Sge': 3600, 'U Aqr': 3600, 'UV Cas': 4800,
               'UW Cen': 3600, 'UX Ant': 3700, 'V CrA': 3600,
               'V1157 Sgr': 3600, 'V1783 Sgr': 3600, 'V2552 Oph': 3600,
               'V2331 Sgr': 3600, 'V3795 Sgr': 3600, 'V391 Sct': 3800,
               'V4017 Sgr': 3600, 'V482 Cyg': 4800, 'V517 Oph': 3600,
               'V739 Sgr': 3600, 'V854 Cen': 3600, 'VZ Sgr': 3600,
               'WX CrA': 3600, 'Y Mus': 3600, 'Z Umi': 4800,
               'HD137613': 3600, 'HD148839': 3600, 'HD173409': 3600,
               'HD175893': 3600, 'HD182040': 3600,
               'HV 5637': 3650, 'W Men': 3600, 'HV 12842': 3650,
               'MACHO-6.6696.60': 4000,
               'MACHO-18.3325.148': 3650, 'MACHO-80.6956.207': 4000,
               'EROS2-LMC-RCB-1': 4000,
               'EROS2-LMC-RCB-2': 3800, 'EROS2-LMC-RCB-3': 3600,
               'EROS2-LMC-RCB-4': 3700, 'EROS2-LMC-RCB-5': 3600,
               'EROS2-LMC-RCB-6': 5800, 'EROS2-SMC-RCB-1': 3800,
               'EROS2-SMC-RCB-2': 3600, 'EROS2-SMC-RCB-3': 4500,
               'J005010.67-694357.7': 3800, 'J005113.58-731036.3': 5500,
               'J054221.91-690259.3': 4600, 'J055643.56-715510.7': 3600,
               'J054123.49-705801.8': 3600, 'B42':3600,
               'A223':3600, 'A249':3600, 'A182':3800,
               'A226':3600, 'A183':3600, 'A166':3600,
               'C17':3600, 'C27':3600, 'F75':3600,
               'C38':3600, 'C20':3800, 'C105':3600,
               'C526':3600, 'A811':3800, 'F152':3600,
               'C539':3800, 'A208':3600,'A977':3800,
               'A980':3600,'C1004':3600,
               'C528':3600,'B564':3600,'B565':3600,
               'B566':3600,'B567':3600,'A798':3600,
               'A770':3600,'B563':3600,'C542':3600,
               'A814':3600,'MACHO-81.8394.1358':4200,'MACHO-80.6956.207':4000,
               'ASAS J050232-7218.9':4000,'J173737.07-072828.1':5900,
               'dLS17':3600,'J181836.38-181732.8':4500,'M37':4000,
               'M38':4250,'P12':4250,'WISE-ToI-5031':3600,'WISE-ToI-150':3600,
               'dLS114':3600,
               'ASAS-DYPer-1':3600,
               'ASAS-DYPer-2':3600,
               'EROS2-LMC-DYPer-1':5500,
               'EROS2-LMC-DYPer-2':5500,
               'EROS2-LMC-DYPer-3':5500,
               'EROS2-LMC-DYPer-4':5500,
               'EROS2-LMC-DYPer-5':5500,
               'EROS2-LMC-DYPer-6':5500,
               'EROS2-LMC-DYPer-7':5500,
               'EROS2-LMC-DYPer-8':5500,
               'EROS2-LMC-DYPer-9':5500,
               'EROS2-LMC-DYPer-10':5500,
               'EROS2-LMC-DYPer-11':5500,
               'EROS2-LMC-DYPer-12':5500,
               'EROS2-LMC-DYPer-13':5500,
               'EROS2-LMC-DYPer-14':5500,
               'EROS2-LMC-DYPer-15':5500,
               'EROS2-LMC-DYPer-16':5500,
               'EROS2-LMC-DYPer-17':5500,
               'EROS2-CG-RCB-2':5500,
               'EROS2-SMC-DYPer-1':5500,
               'EROS2-SMC-DYPer-2':5500,
               'EROS2-SMC-DYPer-3':5500,
               'EROS2-SMC-DYPer-4':5500,
               'EROS2-SMC-DYPer-5':5500,
               'EROS2-SMC-DYPer-6':5500}

def apply_extinction(lam, flux, extinction, name):
    ext = CCM89(Rv=3.1)
    wavelength_w_units = lam*u.AA
    dereddened_flux = np.asarray(flux)/ext.extinguish(wavelength_w_units,Av=extinction)

    #bad_mask = np.where(((flux)/np.max(flux))<0.0001,True,False)
    #if use_alt_mask == True:
    bad_lam_limit = cutoff_dict[name]
        #if np.min(lam) < 3600:
        #    bad_lam_limit = 3600
        #else:
        #    bad_lam_limit = np.min(lam)
    #else:
     #   if name not in ['V2331 Sgr','UW Cen','HE 1015-2050','RZ Nor','EROS2-LMC-RCB-1','EROS2-SMC-RCB-3']:
     #       try:
     #           bad_lam_limit = np.max(lam[bad_mask])
     #       except ValueError:
     #           bad_lam_limit = np.min(lam)
     #   elif name == 'V2331 Sgr':
     #       bad_lam_limit = 4250
     #   else:
     #       bad_lam_limit = np.min(lam)
    good_lam = lam[lam>bad_lam_limit]
    good_flux = flux[lam>bad_lam_limit]
    dereddened_flux_good = dereddened_flux[lam>bad_lam_limit]
    
    return good_lam, dereddened_flux_good

def calculate_eq_width_models_general(model_folder,temperature,lines_list,
                                    center_tolerance=1,width_tolerance=10,continuum_region=50,continuum_width=50):
    folder = model_folder
    eq_width = []
    voigt_fwhm_array = []
    
#     for temperature in temperatures_list:

    os.chdir('/Users/crawford/Desktop/MODELS_VIS_Courtney-Sept2020/')
    temporary = np.loadtxt(all_models[folder][np.str(temperature)],skiprows=3)
    mod_lam = temporary[:,0]
    mod_flux = temporary[:,1]
    mod_flux_smooth = convolve(mod_flux, Box1DKernel(3))
    os.chdir('/Users/crawford/Desktop/')

    ## identify a pseudo continuum around the entire array of lines to measure                                                            
    pseudo_bound_left = lines_list[0]-continuum_width
    pseudo_bound_right = lines_list[-1]+continuum_width

    pseudo_cont_left = mod_flux[np.logical_and(pseudo_bound_left-(continuum_region+continuum_width) < mod_lam,mod_lam < pseudo_bound_left)]
    lam_pseudo_cont_left = mod_lam[np.logical_and(pseudo_bound_left-(continuum_region+continuum_width) < mod_lam,mod_lam < pseudo_bound_left)]
    pseudo_cont_right = mod_flux[np.logical_and(pseudo_bound_right < mod_lam,mod_lam < pseudo_bound_right+(continuum_region+continuum_width))]
    lam_pseudo_cont_right = mod_lam[np.logical_and(pseudo_bound_right < mod_lam,mod_lam < pseudo_bound_right+(continuum_region+continuum_width))]

    ## find the peaks of the pseudo continuum and fit a line through those, weighted by the square of the y value                         
    lam_pseudo_cont = np.concatenate((np.asarray(lam_pseudo_cont_left),np.asarray(lam_pseudo_cont_right)))
    pseudo_cont = np.concatenate((np.asarray(pseudo_cont_left),np.asarray(pseudo_cont_right)))
    peaks, params = find_peaks(pseudo_cont,distance=5)

    p = np.polyfit(lam_pseudo_cont[peaks],pseudo_cont[peaks],1,w=pseudo_cont[peaks]**2)

    lines = lines_list

    for center in lines:   
        ### start by making sure the line is centered where you're searching ###                                                          
        search_region_lam = mod_lam[np.abs(mod_lam-center)<30]
        search_region_flux = -mod_flux[np.abs(mod_lam-center)<30]
        line_bottoms, _ = find_peaks(search_region_flux)
#         plt.plot(search_region_lam,-search_region_flux)                                                                                 
#         plt.plot(search_region_lam[line_bottoms],-search_region_flux[line_bottoms],'x')                                                 
        closest_bottom = np.argmin(np.abs(search_region_lam[line_bottoms]-center))
#         plt.scatter(search_region_lam[line_bottoms][closest_bottom],-search_region_flux[line_bottoms][closest_bottom])                  
#         plt.show()                                                                                                                      
        displacement = search_region_lam[line_bottoms][closest_bottom]-center
        mod_lam = mod_lam - displacement
        print('Shifting wavelength by', displacement, 'for line center', center)

        ## define the line region and some temporary arrays to use
        jumps_left = []
        jumps_right = []
        lam_line_region = mod_lam[np.abs(mod_lam-center)<55]
        flux_line_region = mod_flux_smooth[np.abs(mod_lam-center)<55]

        ## interpolate over the line region, so you can find when it crosses the continuum
        interp_line_region = interpolate.interp1d(lam_line_region,flux_line_region)
        interp_lam = np.linspace(center-50,center+50,1000)
        interp_flux = interp_line_region(interp_lam)

        ## find when the interpolated lines cross the continuum and select regions below that
        refine_lam_line_region = interp_lam[interp_flux < p[0]*interp_lam+p[1]]
        refine_flux_line_region = interp_flux[interp_flux < p[0]*interp_lam+p[1]]

        ## find the "jumps" in the interpolated region and select the closest one as the new line region boundary                         
        ### eventually will want to fix how I'm finding the empty lists
        for i in range(1,len(refine_lam_line_region)):
            if refine_lam_line_region[i]-refine_lam_line_region[i-1] > 0.2 and refine_lam_line_region[i] < center:
                jumps_left.append(i)
            elif refine_lam_line_region[i]-refine_lam_line_region[i-1] > 0.2 and refine_lam_line_region[i] > center:
                jumps_right.append(i)
        if jumps_left == []:
            left_side = refine_lam_line_region[refine_lam_line_region < center]
            flux_left_side = refine_flux_line_region[refine_lam_line_region < center]
            cont_left_side = p[0]*left_side + p[1]
            left_bound = np.argmin(cont_left_side-flux_left_side)
        elif jumps_left != []:
            left_bound = np.max(jumps_left)
        if jumps_right == []:
            right_side = refine_lam_line_region[refine_lam_line_region > center]
            mid_bound = len(refine_lam_line_region)-len(right_side)
            flux_right_side = refine_flux_line_region[refine_lam_line_region > center]
            cont_right_side = p[0]*right_side +p[1]
            right_bound = mid_bound + np.argmin(cont_right_side-flux_right_side)
        elif jumps_right != []:
            right_bound = np.min(jumps_right)
        trim_lam_line_region = refine_lam_line_region[left_bound:right_bound]
        trim_flux_line_region = refine_flux_line_region[left_bound:right_bound]
        cont_line_region = p[0]*trim_lam_line_region + p[1]


        ###########                                                                                                                       
        ## finding the local maxima rather than the derivative turning over                                                               
        ### use trim_lam_line_region, trim_flux_line_region, cont_line_region, flux_region_to_integrate                                   
        left_lam_region_of_line = trim_lam_line_region[trim_lam_line_region < center]
        right_lam_region_of_line = trim_lam_line_region[trim_lam_line_region > center]
        left_flux_region_of_line = trim_flux_line_region[trim_lam_line_region < center]
        right_flux_region_of_line = trim_flux_line_region[trim_lam_line_region > center]
        left_local_maxima, _ = find_peaks(left_flux_region_of_line)
        right_local_maxima, _ = find_peaks(right_flux_region_of_line)
        if len(left_lam_region_of_line[left_local_maxima]) != 0:
            left_bound = np.max(left_lam_region_of_line[left_local_maxima])
        else:
            left_bound = left_lam_region_of_line[0]
        if len(right_lam_region_of_line[right_local_maxima]) != 0:
            right_bound = np.min(right_lam_region_of_line[right_local_maxima])
        else:
            right_bound = right_lam_region_of_line[-1]
        ## find midpoint to define the line region                                                                                        
        mid_point = (left_bound+right_bound)/2
        diff_limit = right_bound - mid_point
        new_lam_line_region = trim_lam_line_region[np.abs(trim_lam_line_region-mid_point) < diff_limit]
        new_flux_line_region = trim_flux_line_region[np.abs(trim_lam_line_region-mid_point) < diff_limit]
        cont_line_region = p[0]*new_lam_line_region + p[1]

        flux_region_to_integrate = (cont_line_region-new_flux_line_region)/cont_line_region
        ###########  


        ## add in voigt profile fitting ##
        try:
            popt, pcov = curve_fit(voigt_fitter,new_lam_line_region,flux_region_to_integrate,
                                   bounds=((center-center_tolerance,-np.inf,0,0),(center+center_tolerance,np.inf,20,20)),
                                   maxfev=15000)
        except ValueError:
            trim_lam_line_region = refine_lam_line_region[left_bound-3:right_bound+3]
            trim_flux_line_region = refine_flux_line_region[left_bound-3:right_bound+3]
            cont_line_region = p[0]*trim_lam_line_region + p[1]
            flux_region_to_integrate = (cont_line_region-trim_flux_line_region)/cont_line_region
            popt, pcov = curve_fit(voigt_fitter,trim_lam_line_region,flux_region_to_integrate,
                                   bounds=((center-center_tolerance,-np.inf,0,0),(center+center_tolerance,np.inf,20,20)),
                                   maxfev=10000)
        #had to increase the max number of evaluations. Check here if issues later
        lam_voigt_profile = np.linspace(center-75,center+75,1000)
        voigt_profile = voigt_fitter(lam_voigt_profile,*popt)
        print(popt)

        voigt_fwhm = 0.5346*popt[2] + np.sqrt(0.2166*(popt[2]**2)+ popt[3]**2)
        print('Voigt FWHM is', voigt_fwhm)

        ## to better visualize the fit itself ##
        cont_voigt = p[0]*lam_voigt_profile + p[1]
        voigt_for_plotting = cont_voigt - voigt_profile*cont_voigt

        area = np.round(np.trapz(voigt_profile,lam_voigt_profile)*u.Angstrom,2)
        if voigt_fwhm > width_tolerance:
            print('Voigt Profile too wide, likely no detection.')
            area = 0*u.AA
        print('Equivalent Width for', center*u.AA, 'for', temperature, 'is:', area)

        plt.plot(mod_lam,mod_flux_smooth,label='original spectrum')
        plt.plot(lam_pseudo_cont,p[0]*lam_pseudo_cont+p[1],label='pseudo-cont')
        plt.plot(lam_line_region,flux_line_region,label='largest line region')
        plt.plot(refine_lam_line_region,refine_flux_line_region,label='below continuum')
        plt.plot(trim_lam_line_region,trim_flux_line_region,linewidth=2,label='trimmed the edges')
#         plt.scatter(left_lam_region_of_line[left_local_maxima],left_flux_region_of_line[left_local_maxima],label='local maxima')        
#         plt.scatter(right_lam_region_of_line[right_local_maxima],right_flux_region_of_line[right_local_maxima],label='local maxima')    
        plt.plot(new_lam_line_region,new_flux_line_region,linewidth=2,label='final changes')
        plt.plot(lam_voigt_profile,voigt_for_plotting,linewidth=2,color='k')
        if len(lines) > 1:
            for i in range(0,len(lines)):
                plt.axvline(lines[i],color='grey')
        plt.axvline(center,color='k')
        plt.xlim(center-100,center+100)
        plt.ylim(-0.5,1.2)
        plt.title(folder + ',' + str(temperature))
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
        
        eq_width.append(area/u.Angstrom)
        voigt_fwhm_array.append(voigt_fwhm)     
    
    return eq_width, voigt_fwhm_array


# def calculate_eq_width_models_singleline(model_folder,lines_list,
#                                         center_tolerance=1,width_tolerance=10,continuum_region=50,continuum_width=50):
#     folder = model_folder
#     temperatures_list = []
#     file_list = os.listdir('/Users/crawford/Desktop/MODELS_VIS_Courtney-Sept2020/' + folder)
#     for i in range(0,len(file_list)):
#         temperatures_list.append(np.int(file_list[i][:4]))
# #     temps[folder] = temperatures_list
#     eq_width = dict()
#     voigt_params = dict()
#     blue_list = []
# #     red_list = []
# #     center_list = []
# #     voigt_red = []
#     voigt_blue = []
# #     voigt_center = []
#     voigt_fwhm_array = []
    
#     for temperature in temperatures_list:

#         os.chdir('/Users/crawford/Desktop/MODELS_VIS_Courtney-Sept2020/')
#         temporary = np.loadtxt(all_models[folder][np.str(temperature)],skiprows=3)
#         mod_lam = temporary[:,0]
#         mod_flux = temporary[:,1]
#         mod_flux_smooth = convolve(mod_flux, Box1DKernel(3))
#         os.chdir('/Users/crawford/Desktop/')

#         pseudo_bound_left = lines_list[0]-continuum_width
#         pseudo_bound_right = lines_list[-1]+continuum_width

#         pseudo_cont_left = mod_flux[np.logical_and(pseudo_bound_left-(continuum_region+continuum_width) < mod_lam,mod_lam < pseudo_bound_left)]
#         lam_pseudo_cont_left = mod_lam[np.logical_and(pseudo_bound_left-(continuum_region+continuum_width) < mod_lam,mod_lam < pseudo_bound_left)]
#         pseudo_cont_right = mod_flux[np.logical_and(pseudo_bound_right < mod_lam,mod_lam < pseudo_bound_right+(continuum_region+continuum_width))]
#         lam_pseudo_cont_right = mod_lam[np.logical_and(pseudo_bound_right < mod_lam,mod_lam < pseudo_bound_right+(continuum_region+continuum_width))]

#         ## find the peaks of the pseudo continuum and fit a line through those, weighted by the square of the y value
#         lam_pseudo_cont = np.concatenate((np.asarray(lam_pseudo_cont_left),np.asarray(lam_pseudo_cont_right)))
#         pseudo_cont = np.concatenate((np.asarray(pseudo_cont_left),np.asarray(pseudo_cont_right)))
#         peaks, params = find_peaks(pseudo_cont,distance=5)

#         p = np.polyfit(lam_pseudo_cont[peaks],pseudo_cont[peaks],1,w=pseudo_cont[peaks]**2)

        

#         lines = lines_list

#         for center in lines:
            
#             ### start by making sure the line is centered where you're searching ###
#             search_region_lam = mod_lam[np.abs(mod_lam-center)<30]
#             search_region_flux = -mod_flux[np.abs(mod_lam-center)<30]
#             line_bottoms, _ = find_peaks(search_region_flux)
#             closest_bottom = np.argmin(np.abs(search_region_lam[line_bottoms]-center))
#             displacement = search_region_lam[line_bottoms][closest_bottom]-center
#             mod_lam = mod_lam - displacement 
#             print('Shifting wavelength by', displacement, 'for line center', center)
            
#             ## define the line region and some temporary arrays to use  
#             jumps_left = []
#             jumps_right = []
#             lam_line_region = mod_lam[np.abs(mod_lam-center)<55]
#             flux_line_region = mod_flux_smooth[np.abs(mod_lam-center)<55]
            
#             ## interpolate over the line region, so you can find when it crosses the continuum 
#             interp_line_region = interpolate.interp1d(lam_line_region,flux_line_region)
#             interp_lam = np.linspace(center-50,center+50,1000)
#             interp_flux = interp_line_region(interp_lam)

#             ## find when the interpolated lines cross the continuum and select regions below that 
#             refine_lam_line_region = interp_lam[interp_flux < p[0]*interp_lam+p[1]]
#             refine_flux_line_region = interp_flux[interp_flux < p[0]*interp_lam+p[1]]
            
#             ## find the "jumps" in the interpolated region and select the closest one as the new line region boundary                         
#             ### eventually will want to fix how I'm finding the empty lists 
#             for i in range(1,len(refine_lam_line_region)):
#                 if refine_lam_line_region[i]-refine_lam_line_region[i-1] > 0.2 and refine_lam_line_region[i] < center:
#                     jumps_left.append(i)
#                 elif refine_lam_line_region[i]-refine_lam_line_region[i-1] > 0.2 and refine_lam_line_region[i] > center:
#                     jumps_right.append(i)
#             if jumps_left == []:
#                 left_side = refine_lam_line_region[refine_lam_line_region < center]
#                 flux_left_side = refine_flux_line_region[refine_lam_line_region < center]
#                 cont_left_side = p[0]*left_side + p[1]
#                 left_bound = np.argmin(cont_left_side-flux_left_side)
#             elif jumps_left != []:
#                 left_bound = np.max(jumps_left)
#             if jumps_right == []:
#                 right_side = refine_lam_line_region[refine_lam_line_region > center]
#                 mid_bound = len(refine_lam_line_region)-len(right_side)
#                 flux_right_side = refine_flux_line_region[refine_lam_line_region > center]
#                 cont_right_side = p[0]*right_side +p[1]
#                 right_bound = mid_bound + np.argmin(cont_right_side-flux_right_side)
#             elif jumps_right != []:
#                 right_bound = np.min(jumps_right)
#             trim_lam_line_region = refine_lam_line_region[left_bound:right_bound]
#             trim_flux_line_region = refine_flux_line_region[left_bound:right_bound]
#             cont_line_region = p[0]*trim_lam_line_region + p[1]

            
#             ###########                                                                                                                       
#             ## finding the local maxima rather than the derivative turning over                                                               
#             ### use trim_lam_line_region, trim_flux_line_region, cont_line_region, flux_region_to_integrate                                   
#             left_lam_region_of_line = trim_lam_line_region[trim_lam_line_region < center]
#             right_lam_region_of_line = trim_lam_line_region[trim_lam_line_region > center]
#             left_flux_region_of_line = trim_flux_line_region[trim_lam_line_region < center]
#             right_flux_region_of_line = trim_flux_line_region[trim_lam_line_region > center]
#             left_local_maxima, _ = find_peaks(left_flux_region_of_line)
#             right_local_maxima, _ = find_peaks(right_flux_region_of_line)
#             if len(left_lam_region_of_line[left_local_maxima]) != 0:
#                 left_bound = np.max(left_lam_region_of_line[left_local_maxima])
#             else:
#                 left_bound = left_lam_region_of_line[0]
#             if len(right_lam_region_of_line[right_local_maxima]) != 0:
#                 right_bound = np.min(right_lam_region_of_line[right_local_maxima])
#             else:
#                 right_bound = right_lam_region_of_line[-1]
#             ## find midpoint to define the line region                                                                                        
#             mid_point = (left_bound+right_bound)/2
#             diff_limit = right_bound - mid_point
#             new_lam_line_region = trim_lam_line_region[np.abs(trim_lam_line_region-mid_point) < diff_limit]
#             new_flux_line_region = trim_flux_line_region[np.abs(trim_lam_line_region-mid_point) < diff_limit]
#             cont_line_region = p[0]*new_lam_line_region + p[1]

#             flux_region_to_integrate = (cont_line_region-new_flux_line_region)/cont_line_region
#             ########### 

            
#             ## add in voigt profile fitting ##
#             try:
#                 popt, pcov = curve_fit(voigt_fitter,new_lam_line_region,flux_region_to_integrate,
#                                    bounds=((center-center_tolerance,-np.inf,0,0),(center+center_tolerance,np.inf,20,20)),
#                                    maxfev=15000)
#             except ValueError:
#                 trim_lam_line_region = refine_lam_line_region[left_bound-3:right_bound+3]
#                 trim_flux_line_region = refine_flux_line_region[left_bound-3:right_bound+3]
#                 cont_line_region = p[0]*trim_lam_line_region + p[1]
#                 flux_region_to_integrate = (cont_line_region-trim_flux_line_region)/cont_line_region
#                 popt, pcov = curve_fit(voigt_fitter,trim_lam_line_region,flux_region_to_integrate,
#                                        bounds=((center-center_tolerance,-np.inf,0,0),(center+center_tolerance,np.inf,20,20)),
#                                        maxfev=10000)
#             #had to increase the max number of evaluations. Check here if issues later
#             lam_voigt_profile = np.linspace(center-75,center+75,1000)
#             voigt_profile = voigt_fitter(lam_voigt_profile,*popt)
#             print(popt)
            
#             voigt_fwhm = 0.5346*popt[2] + np.sqrt(0.2166*(popt[2]**2)+ popt[3]**2)
#             print('Voigt FWHM is', voigt_fwhm)
            
#             ## to better visualize the fit itself ##
#             cont_voigt = p[0]*lam_voigt_profile + p[1]
#             voigt_for_plotting = cont_voigt - voigt_profile*cont_voigt

#             area = np.round(np.trapz(voigt_profile,lam_voigt_profile)*u.Angstrom,2)
#             if voigt_fwhm > width_tolerance:
#                 print('Voigt Profile too wide, likely no detection.')
#                 area = 0*u.AA
#             print('Equivalent Width for', center*u.AA, 'for', name, 'is:', area)

#             plt.plot(mod_lam,mod_flux_smooth,label='original spectrum')
#             plt.plot(lam_pseudo_cont,p[0]*lam_pseudo_cont+p[1],label='pseudo-cont')
#             plt.plot(lam_line_region,flux_line_region,label='largest line region')
#             plt.plot(refine_lam_line_region,refine_flux_line_region,label='below continuum')
#             plt.plot(trim_lam_line_region,trim_flux_line_region,linewidth=2,label='trimmed the edges')
#     #         plt.scatter(left_lam_region_of_line[left_local_maxima],left_flux_region_of_line[left_local_maxima],label='local maxima')        
#     #         plt.scatter(right_lam_region_of_line[right_local_maxima],right_flux_region_of_line[right_local_maxima],label='local maxima')    
#             plt.plot(new_lam_line_region,new_flux_line_region,linewidth=2,label='final changes')
#             plt.plot(lam_voigt_profile,voigt_for_plotting,linewidth=2,color='k')
#     #         plt.axvline(center,color='k')                                                                                                   
#             if len(lines) > 1:
#                 for i in range(0,len(lines)):
#                     plt.axvline(lines[i],color='grey')
#             plt.axvline(center,color='k')
#     #         plt.xlim(center-50,center+50)                                                                                                   
#             plt.xlim(center-100,center+100)
#             plt.ylim(-0.5,1.2)
#             plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
#             plt.title(folder + ',' + str(temperature))
#             plt.show()

#             if center == lines[0]:
#                 blue_list.append(area/u.Angstrom)
#                 voigt_blue.append(popt)
#                 voigt_fwhm_array.append(voigt_fwhm)
        
        
#     eq_width = blue_list
#     voigt_params = voigt_blue
    
#     return eq_width, voigt_params, temperatures_list, voigt_fwhm_array


def calculate_eq_width_observations(wavelength_array,flux_array,star_name,lines_list,
                                    center_tolerance=1,width_tolerance=10,continuum_region=50,continuum_width=50):
    ## initialize some variables from inputs
    lam = wavelength_array
    flux = flux_array
    name = star_name
    eq_width = []
    voigt_fwhm_array = []
    
    ## identify a pseudo continuum around the entire array of lines to measure
    pseudo_bound_left = lines_list[0]-continuum_width
    pseudo_bound_right = lines_list[-1]+continuum_width
    
#     print(min(lam),pseudo_bound_left,pseudo_bound_left-(continuum_region+continuum_width))
#     print(len(lam),len(flux))
    
    pseudo_cont_left = flux[np.logical_and(pseudo_bound_left-(continuum_region+continuum_width) < lam,lam < pseudo_bound_left)]
    lam_pseudo_cont_left = lam[np.logical_and(pseudo_bound_left-(continuum_region+continuum_width) < lam,lam < pseudo_bound_left)]
    pseudo_cont_right = flux[np.logical_and(pseudo_bound_right < lam,lam < pseudo_bound_right+(continuum_region+continuum_width))]
    lam_pseudo_cont_right = lam[np.logical_and(pseudo_bound_right < lam,lam < pseudo_bound_right+(continuum_region+continuum_width))]

    ## find the peaks of the pseudo continuum and fit a line through those, weighted by the square of the y value
    lam_pseudo_cont = np.concatenate((np.asarray(lam_pseudo_cont_left),np.asarray(lam_pseudo_cont_right)))
    pseudo_cont = np.concatenate((np.asarray(pseudo_cont_left),np.asarray(pseudo_cont_right)))
    peaks, params = find_peaks(pseudo_cont,distance=5)

    p = np.polyfit(lam_pseudo_cont[peaks],pseudo_cont[peaks],1,w=pseudo_cont[peaks]**2)

    lines = lines_list
    ## measure each line individually
    for center in lines:
        
        ### start by making sure the line is centered where you're searching ###
        search_region_lam = lam[np.abs(lam-center)<30]
        search_region_flux = -flux[np.abs(lam-center)<30]
        line_bottoms, _ = find_peaks(search_region_flux)
#         plt.plot(search_region_lam,-search_region_flux)
#         plt.plot(search_region_lam[line_bottoms],-search_region_flux[line_bottoms],'x')
        closest_bottom = np.argmin(np.abs(search_region_lam[line_bottoms]-center))
#         plt.scatter(search_region_lam[line_bottoms][closest_bottom],-search_region_flux[line_bottoms][closest_bottom])
#         plt.show()
        displacement = search_region_lam[line_bottoms][closest_bottom]-center
        lam = lam - displacement 
        print('Shifting wavelength by', displacement, 'for line center', center)
        
        ## define the line region and some temporary arrays to use
        jumps_left = []
        jumps_right = []
        lam_line_region = lam[np.abs(lam-center)<55]
        flux_line_region = flux[np.abs(lam-center)<55]

        ## interpolate over the line region, so you can find when it crosses the continuum
        interp_line_region = interpolate.interp1d(lam_line_region,flux_line_region)
        interp_lam = np.linspace(center-50,center+50,1000)
        interp_flux = interp_line_region(interp_lam)


        ## find when the interpolated lines cross the continuum and select regions below that
        refine_lam_line_region = interp_lam[interp_flux < p[0]*interp_lam+p[1]]
        refine_flux_line_region = interp_flux[interp_flux < p[0]*interp_lam+p[1]]
        ## find the "jumps" in the interpolated region and select the closest one as the new line region boundary
        ### eventually will want to fix how I'm finding the empty lists
        for i in range(1,len(refine_lam_line_region)):
            if refine_lam_line_region[i]-refine_lam_line_region[i-1] > 0.2 and refine_lam_line_region[i] < center:
        #         print('jump', i)
                jumps_left.append(i)
            elif refine_lam_line_region[i]-refine_lam_line_region[i-1] > 0.2 and refine_lam_line_region[i] > center:
                jumps_right.append(i)
        # print(jumps_left,jumps_right)
        if jumps_left == []:
            left_side = refine_lam_line_region[refine_lam_line_region < center]
            flux_left_side = refine_flux_line_region[refine_lam_line_region < center]
            cont_left_side = p[0]*left_side + p[1]
            left_bound = np.argmin(cont_left_side-flux_left_side)
    #             left_bound = np.argmax(left_side)
        elif jumps_left != []:
            left_bound = np.max(jumps_left)
        if jumps_right == []:
            right_side = refine_lam_line_region[refine_lam_line_region > center]
            mid_bound = len(refine_lam_line_region)-len(right_side)
            flux_right_side = refine_flux_line_region[refine_lam_line_region > center]
            cont_right_side = p[0]*right_side +p[1]
            right_bound = mid_bound + np.argmin(cont_right_side-flux_right_side)
    #             print(cont_right_side-flux_right_side)
    #             print(flux_right_side)

    #             right_bound = np.argmax(right_side)
        elif jumps_right != []:
            right_bound = np.min(jumps_right)
        trim_lam_line_region = refine_lam_line_region[left_bound:right_bound]
        trim_flux_line_region = refine_flux_line_region[left_bound:right_bound]
        cont_line_region = p[0]*trim_lam_line_region + p[1]


        ###########
        ## finding the local maxima rather than the derivative turning over
        ### use trim_lam_line_region, trim_flux_line_region, cont_line_region, flux_region_to_integrate
        left_lam_region_of_line = trim_lam_line_region[trim_lam_line_region < center]
        right_lam_region_of_line = trim_lam_line_region[trim_lam_line_region > center]
        left_flux_region_of_line = trim_flux_line_region[trim_lam_line_region < center]
        right_flux_region_of_line = trim_flux_line_region[trim_lam_line_region > center]
        left_local_maxima, _ = find_peaks(left_flux_region_of_line)
        right_local_maxima, _ = find_peaks(right_flux_region_of_line)
        if len(left_lam_region_of_line[left_local_maxima]) != 0:
            left_bound = np.max(left_lam_region_of_line[left_local_maxima])
        else:
            left_bound = left_lam_region_of_line[0]
        if len(right_lam_region_of_line[right_local_maxima]) != 0:
            right_bound = np.min(right_lam_region_of_line[right_local_maxima])
        else:
            right_bound = right_lam_region_of_line[-1]        
        ## find midpoint to define the line region
        mid_point = (left_bound+right_bound)/2
        diff_limit = right_bound - mid_point
        new_lam_line_region = trim_lam_line_region[np.abs(trim_lam_line_region-mid_point) < diff_limit]
        new_flux_line_region = trim_flux_line_region[np.abs(trim_lam_line_region-mid_point) < diff_limit]
        cont_line_region = p[0]*new_lam_line_region + p[1]

        flux_region_to_integrate = (cont_line_region-new_flux_line_region)/cont_line_region    
        ###########



        ## add in voigt profile fitting ##
        try:
            popt, pcov = curve_fit(voigt_fitter,new_lam_line_region,flux_region_to_integrate,
                                   bounds=((center-center_tolerance,-np.inf,0,0),(center+center_tolerance,np.inf,20,20)), 
                                   maxfev=15000) 
        except ValueError:
            trim_lam_line_region = refine_lam_line_region[left_bound-3:right_bound+3]
            trim_flux_line_region = refine_flux_line_region[left_bound-3:right_bound+3]
            cont_line_region = p[0]*trim_lam_line_region + p[1]
            flux_region_to_integrate = (cont_line_region-trim_flux_line_region)/cont_line_region
            popt, pcov = curve_fit(voigt_fitter,trim_lam_line_region,flux_region_to_integrate,
                                   bounds=((center-center_tolerance,-np.inf,0,0),(center+center_tolerance,np.inf,20,20)), 
                                   maxfev=10000) 
        #had to increase the max number of evaluations. Check here if issues later
        lam_voigt_profile = np.linspace(center-75,center+75,1000)
        voigt_profile = voigt_fitter(lam_voigt_profile,*popt)
        print(popt)
#         print(pcov[0,0],pcov[1,1],pcov[2,2],pcov[3,3])
#         error_value = np.sqrt(pcov[0,0]**2+pcov[1,1]**2+pcov[2,2]**2+pcov[3,3]**2)
#         print(error_value)
        
        voigt_fwhm = 0.5346*popt[2] + np.sqrt(0.2166*(popt[2]**2)+ popt[3]**2)
        print('Voigt FWHM is', voigt_fwhm)

        ## to better visualize the fit itself ##
        cont_voigt = p[0]*lam_voigt_profile + p[1]
        voigt_for_plotting = cont_voigt - voigt_profile*cont_voigt

        area = np.round(np.trapz(voigt_profile,lam_voigt_profile)*u.Angstrom,2)
        if voigt_fwhm > width_tolerance:
            print('Voigt Profile too wide, likely no detection.')
            area = 0*u.AA
        print('Equivalent Width for', center*u.AA, 'for', name, 'is:', area)
        ##########


        plt.plot(lam,flux,label='original spectrum')
        plt.plot(lam_pseudo_cont,p[0]*lam_pseudo_cont+p[1],label='pseudo-cont')
        plt.plot(lam_line_region,flux_line_region,label='largest line region')
        plt.plot(refine_lam_line_region,refine_flux_line_region,label='below continuum')
        plt.plot(trim_lam_line_region,trim_flux_line_region,linewidth=2,label='trimmed the edges')
#         plt.scatter(left_lam_region_of_line[left_local_maxima],left_flux_region_of_line[left_local_maxima],label='local maxima')
#         plt.scatter(right_lam_region_of_line[right_local_maxima],right_flux_region_of_line[right_local_maxima],label='local maxima')
        plt.plot(new_lam_line_region,new_flux_line_region,linewidth=2,label='final changes')
        plt.plot(lam_voigt_profile,voigt_for_plotting,linewidth=2,color='k')
#         plt.axvline(center,color='k')
        if len(lines) > 1:
            for i in range(0,len(lines)):
                plt.axvline(lines[i],color='grey')
        plt.axvline(center,color='k')
#         plt.xlim(center-50,center+50)
        plt.xlim(center-100,center+100)
        plt.title(name)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.ylim(-0.2,1.2)
        plt.show()

        eq_width.append(area/u.Angstrom)
        voigt_fwhm_array.append(voigt_fwhm)
        
    return eq_width, voigt_fwhm_array


### probably can only measure one band at a time
def calculate_band_eq_width_observations(wavelength_array,flux_array,star_name,band_region,
                                         continuum_region_left,continuum_region_right):    
    ## initialize some variables from inputs
    lam = wavelength_array
    flux = flux_array
    name = star_name
    eq_width = []
#     voigt_fwhm_array = []
    
    ## identify a pseudo continuum around the entire array of lines to measure
    #### use the user-defined regions lists
    pseudo_cont_left = flux[(lam <= continuum_region_left[-1]) & (lam >= continuum_region_left[0])]
    lam_pseudo_cont_left = lam[(lam <= continuum_region_left[-1]) & (lam >= continuum_region_left[0])]
    pseudo_cont_right = flux[(lam <= continuum_region_right[-1]) & (lam >= continuum_region_right[0])]
    lam_pseudo_cont_right = lam[(lam <= continuum_region_right[-1]) & (lam >= continuum_region_right[0])]
    
    pseudo_cont = np.concatenate((np.asarray(pseudo_cont_left),np.asarray(pseudo_cont_right)))
    lam_pseudo_cont = np.concatenate((np.asarray(lam_pseudo_cont_left),np.asarray(lam_pseudo_cont_right)))
    
    ## fit a line through the pseudo-continuum
    p = np.polyfit(lam_pseudo_cont,pseudo_cont,1)
    
    
    ## define the line/band region
    lam_line_region = lam[(lam <= band_region[-1]) & (lam >= band_region[0])]
    flux_line_region = flux[(lam <= band_region[-1]) & (lam >= band_region[0])]
    
    cont_line_region = p[0]*lam_line_region + p[1]
    flux_region_to_integrate = (cont_line_region-flux_line_region)/cont_line_region 
    
#     area = np.round(np.trapz(voigt_profile,lam_voigt_profile),2)
    
    eq_width = np.round(np.trapz(flux_region_to_integrate,lam_line_region),2)
    if eq_width < 0:
        eq_width = 0
    print('Equivalent Width for', band_region, 'for', name, 'is:', eq_width)
    
    index_mag = -2.5*np.log10((1/(lam_line_region[-1]-lam_line_region[0]))*
                              np.trapz(flux_line_region/cont_line_region))
    print('Magnitude Index for', band_region, 'for', name, 'is:', index_mag)
    
    plt.plot(lam,flux,label='spectrum')
    plt.plot(lam_pseudo_cont,p[0]*lam_pseudo_cont+p[1],label='pseudo cont')
    plt.plot(lam_pseudo_cont_left,pseudo_cont_left,linewidth=2,label='pseudo cont region')
    plt.plot(lam_pseudo_cont_right,pseudo_cont_right,color='tab:green',linewidth=2)
    plt.plot(lam_line_region,flux_line_region,label='line region')
    plt.xlim(continuum_region_left[0]-100,continuum_region_right[-1]+100)
    plt.title(name)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylim(-0.2,1.2)
    plt.show()
        
    return eq_width, index_mag
