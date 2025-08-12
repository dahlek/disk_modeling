# https://github.com/csalyk/spectools_ir/blob/main/docs/example.ipynb

import sys
#sys.path.append('/Users/alexaanderson/opt/miniconda3/lib/python3.8/site-packages/')
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
import pickle as pickle
import os as os
import pandas as pd

from spectools_ir.utils import extract_hitran_data, spec_convol, make_rotation_diagram, get_molmass
from spectools_ir.utils import compute_thermal_velocity, sigma_to_fwhm, fwhm_to_sigma, wn_to_k, spec_convol_R
from spectools_ir.utils import get_miri_mrs_resolution, get_miri_mrs_wavelengths, make_miri_mrs_figure

from spectools_ir.flux_calculator import calc_fluxes, make_lineshape

from spectools_ir.slabspec import make_spec

from spectools_ir.slab_fitter import Config, LineData,Retrieval
from spectools_ir.slab_fitter import corner_plot, trace_plot, find_best_fit, compute_model_fluxes
from spectools_ir.slab_fitter import calc_solid_angle, calc_radius
from spectools_ir.slab_fitter import read_data_from_file, get_samples

from astropy.table import Table, vstack, QTable
import math
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.constants import c
from astropy import units as un

def chisq(data,model):
    return np.sum((data-model)**2/model)

def gaussian(x, a, b, c, d):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

#DEFINE COMPOSITE MODEL = DATA * CONSTANT
def model(f, c):
    fnew = f * c
    return fnew

def readin(fname):
    '''
    Parameters
    ---------
    fname : string
        path and filename

    Returns
    ---------
    wl: wavelength in microns
    flux: flux in normalized units
    uflux: flux uncertainty in normalized units
    wl_min: minimum wavelength of data in microns
    wl_max: maximum wavelength of data in microns
    wl_min_intermediate: maximum wavelength of third order data
    wl_max_intermediate: minimum wavelength of second order data
    '''

    #READ IN NIRSPEC DATA
    infile=Table.read(fname, format='ascii')

    #CHANGE TO PANDAS
    df=infile.to_pandas()

    #RETRIEVE WAVE, FLUX, AND UFLUX VARIABLES
    wl = df['wave']
    flux = df['flux']
    flux_error = df['uflux']

    #FIND MINIMUM AND MAXIMUM WAVELENGTHS IN SPEC
    wl_min = np.min(wl); wl_max = np.max(wl)

    #SPLIT BY ORDER
    f = wl_min
    for i in wl:
        if i-f > 0.0003:
            wl_max_intermediate = i
            wl_min_intermediate = f

        f = np.copy(i)

    #RE-READ IN DATA
    infile=Table.read(fname, format='ascii')
    df=infile.to_pandas()
    wl_min = np.min(wl); wl_max = np.max(wl)

    #DROP LOCATIONS WHERE NEGATIVE FLUX IS REPORTED
    # !! changed to setting these values to nan; created an issue later on in transitionfinder() with missing indexes
    #wl = df.drop(np.where(flux < 0)[0])['wave']
    flux[np.where(flux < 0)[0]] = np.nan
    flux_error[np.where(flux < 0)[0]] = np.nan

    return wl, flux, flux_error, wl_min, wl_max, wl_min_intermediate, wl_max_intermediate

def fullplot(wl, flux, flux_error, wl_min, wl_max, wl_min_intermediate, wl_max_intermediate, hitran_data):
    '''
    Parameters
    ---------
    wl: wavelength in microns
    flux: flux in normalized units
    uflux: flux uncertainty in normalized units
    wl_min: minimum wavelength of data in microns
    wl_max: maximum wavelength of data in microns
    wl_min_intermediate: maximum wavelength of third order data
    wl_max_intermediate: minimum wavelength of second order data

    Returns
    ---------
    line_ids: indices of HITRAN table lines within wavelength range specified by inputs
    '''
    fig=plt.figure(figsize=(12,6))

    line_ids = []

    ax1=fig.add_subplot(211)
    ax1.plot(wl,flux)
    ax1.set_xlim(wl_min,wl_min_intermediate)
    for i,mywave in enumerate(hitran_data['wave']):
        if( (mywave>wl_min) & (mywave<wl_min_intermediate) ):
            ax1.axvline(mywave,color='C1')
            ax1.text(hitran_data['wave'][i],1.4,hitran_data['Qpp'][i].strip())
            line_ids.append(i)
    ax1.set_ylabel('Flux [Jy]',fontsize=14)

    ax2=fig.add_subplot(212)
    ax2.plot(wl,flux)
    ax2.set_xlim(wl_max_intermediate,wl_max)
    for i,mywave in enumerate(hitran_data['wave']):
        if( (mywave>wl_max_intermediate) & (mywave<wl_max) ):
            ax2.axvline(mywave,color='C1')
            ax2.text(hitran_data['wave'][i],1.4,hitran_data['Qpp'][i].strip())
            line_ids.append(i)
    ax2.set_xlabel('Wavelength [$\mu$m]',fontsize=14)
    ax2.set_ylabel('Flux [Jy]',fontsize=14)

    return line_ids

def transitionfinder(wl, flux, flux_error, wl_min, wl_max, line_ids, hitran_data, plot_query=0, delta_wl=0.004, thresh=0.4):
    # upgraded version

    # plot out individual lines and print difference between min/max flux location and hitran wavelength to check for offset
    # contains an option to check individual lines (line_indices_to_check when defined as a list of indicies)
    # added list that will save the names of transitions w/ too little flux (amount of flux determined by flux_threshold_percent)

    absorption = 0; emission = 1 # switches for looking for min or max flux to identify the peak
    flux_threshold_percent = thresh # will reject lines if there's this % of datapoints missing from +/- this region.
    flux_threshold_percent_fwhm = thresh # will reject line if less than this % of datapoints missing from the line region (within fwhm_guesstimation)
    fwhm_guesstimation = 0.001 # will reject if the line +/- this region is missing more than flux_threshold_percent of its datapoints.
    #Can also probe a line (as opposed to continuum) better by shrinking the wavelength window (delta_wl)?
    #delta_wl = 0.004 # wavelength window, centered around hitran line wavlength

    wl_offset = []
    reject_list = [] # will save hitran transition labels for any lines rejected
    # !! maybe make this a general % list, and make another list for lines that lack points in the line itself?
    wl_list = []
    trans_list = []

    line_indices_to_check = np.copy(line_ids) # will look at all loaded Hitran lines

    list_of_illegal_transitions = ['R  1', 'R  0', 'P  1', 'P  2', 'P  3', 'P  4']

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    fig=plt.figure(figsize=(6,6))

    for i,mywave in enumerate(hitran_data['wave']):
        if hitran_data['Qpp'][i].strip() in list_of_illegal_transitions:
            print('Low J trans here')
            pass
        else:
            if i in line_indices_to_check:
                if( (mywave>wl_min) & (mywave<wl_max) ): # if  pull the wavelengths in between delta_wl
                    if absorption == 1 and emission == 0: # find peak using minimum
                        measured_flux = np.min(flux[np.where((hitran_data['wave'][i]-delta_wl < wl) & (hitran_data['wave'][i]+delta_wl > wl))[0]])
                    elif emission == 1 and absorption == 0: # find peak using maximum
                        print(i)
                        measured_flux = np.max(flux[np.where((hitran_data['wave'][i]-delta_wl < wl) & (hitran_data['wave'][i]+delta_wl > wl))[0]])
                    print('Measured line appears to be at:',measured_flux,'Jy')
                    if plot_query == 1:
                        # plot line +/- delta_wl
                        plt.plot(wl,flux)
                        plt.ylabel('Flux (Jy)')
                        plt.xlabel('Wavelength (microns)')
                        plt.xlim(hitran_data['wave'][i]-delta_wl,hitran_data['wave'][i]+delta_wl)
                        plt.axvline(mywave,color='C1')
                        plt.plot([(hitran_data['wave'][i]-fwhm_guesstimation/2),(hitran_data['wave'][i]+fwhm_guesstimation/2)],\
                             [measured_flux/2, measured_flux/2],color='red',label='FWHM guesstimation')
                        plt.text(hitran_data['wave'][i],1.4,hitran_data['Qpp'][i].strip())
                        plt.show()

                    # indices corresponding to plot location, for whole window and for 2*fwhm
                    index_range = np.where((hitran_data['wave'][i]-delta_wl < wl) & (hitran_data['wave'][i]+delta_wl > wl))[0]
                    index_range_fwhm = np.where((hitran_data['wave'][i]-(fwhm_guesstimation/2) < wl) & (hitran_data['wave'][i]+(fwhm_guesstimation/2) > wl))[0]

                    number_of_flux_measurements = len(np.where(flux[index_range].notna())[0] == True) # places without nans within the plotted range
                    number_of_flux_measurements_fwhm = len(np.where(flux[index_range_fwhm].notna())[0] == True)

                    total_number_flux_measurements = len(flux[index_range])
                    fwhm_number_flux_measurements = len(flux[index_range_fwhm])

                    percent_flux = number_of_flux_measurements/total_number_flux_measurements # % of populated datapoints
                    percent_flux_fwhm = number_of_flux_measurements_fwhm/fwhm_number_flux_measurements # % of populated datapoints within fwhm window

                    if percent_flux < flux_threshold_percent: # if there's only a few flux data points, ignore
                        print('!! Ignoring this transition due to lack of total available datapoints (less than '+str(flux_threshold_percent*100)+'%) !!')
                        reject_list.append((hitran_data['wave'][i], hitran_data['Qpp'][i].strip()))
                        continue

                    elif percent_flux_fwhm < flux_threshold_percent_fwhm: # if the line isn't as populated
                        print('!! Ignoring this transition due to lack of available *line* datapoints (less than '+str(flux_threshold_percent_fwhm*100)+'%) !!')
                        reject_list.append((hitran_data['wave'][i], hitran_data['Qpp'][i].strip()))
                        continue

                    # find wavelength offset
                    print('Difference:')
                    print('Measured flux wl:',wl[np.where(flux == measured_flux)[0][0]])
                    print('Hitran line wl:',hitran_data['wave'][i])
                    print(wl[np.where(flux == measured_flux)[0][0]]-hitran_data['wave'][i])
                    wl_offset.append(wl[np.where(flux == measured_flux)[0][0]]-hitran_data['wave'][i])

                    #APPEND ACTUAL LINE CENTER LOCATION TO LIST OUTSIDE OF LOOP
                    wl_list.append(wl[np.where(flux == measured_flux)[0][0]])
                    trans_list.append(hitran_data['Qpp'][i].strip())

    return wl_list, trans_list, reject_list

def energy_separator_wl(wl_list, trans_list):
    '''
    Parameters
    ---------
    wl_list: list of identified transition wavelengths
    trans_list: list of identified transition names

    Returns
    ---------
    lowj: astropy table of low-energy (P5 - P18) transition wavelengths
    highj: astropy table of high-energy (P > P18) transition wavelengths
    lowjtrans: astropy table of low-energy transition names
    highjtrans: astropy table of high-energy transition names
    '''

    #CREATES ASTROPY TABLE OF WAVELENGTHS
    newtab = Table([wl_list], names=['wave'])

    #DEFINE MASK TO CAPTURE TRANSITIONS FROM P5 - P18
    m1 = newtab['wave'] < 4.9
    m2 = newtab['wave'] > 4.7
    m = m1 & m2

    #FILTER LOW-J AND HIGH-J TRANSITION WAVELENGTHS BASED ON MASK
    lowj = newtab[m]
    highj = newtab[newtab['wave'] > 4.9]

    #PERFORM SAME OPERATION FOR TRANSITION NAMES
    transtab = Table([trans_list], names=['wave'])
    lowjtrans = transtab[m]
    highjtrans = transtab[newtab['wave'] > 4.9]

    return lowj, highj, lowjtrans, highjtrans

def energy_separator_transition(w_list, t_list, cutoff=30):
    '''
    Parameters
    ---------
    wl_list: list of identified transition wavelengths
    trans_list: list of identified transition names
    cutoff: int, some J-value which dictates the high and low energy cutoff

    Returns
    ---------
    lowj: list of low-energy (P5 - P18) transition wavelengths
    highj: list of high-energy (P > P18) transition wavelengths
    lowjtrans: list of low-energy transition names
    highjtrans: list of high-energy transition names
    '''
    lowen = []; lowenname = [] # low energy wavelengths and transition names
    highen = []; highenname = [] # low energy wavelengths and transition names
    
    for i,transition in enumerate(t_list):
        J_value = int(t_list[i][2:]) # or is  it J-1?
        if J_value >= cutoff:
            highen.append(w_list[i])
            highenname.append(t_list[i])
        elif J_value < cutoff:
            lowen.append(w_list[i])
            lowenname.append(t_list[i])
    
    return lowen, highen, lowenname, highenname


def make_lineshape_new(wave,flux,uflux, lineflux_data, dv=3., voffset=None,norm=None):
    '''

    Parameters
    ---------
    wave : numpy array
        set of wavelengths for spectrum, in units of microns
    flux : numpy array
        set of fluxes for spectrum, in units of Jy
    lineflux_data : astropy table
        table in same format as flux_calculator output
    dv : float, optional
        bin size for resultant lineshape, in km/s.  Defaults to 3 km/s.
    voffset : float, optional
        Doppler shift of observed spectrum in km/s.  Defaults to median of lineflux fits.
    norm : str, optional
        String describing normalization type.  Currently only option is 'Maxmin', which sets max to 1, min to 0.  Defaults to None.

    Returns
    ---------
    (interpvel,interpflux): tuple containing interpolated line shape

    '''
    w0=np.array(lineflux_data['wave'])

    nlines=np.size(w0)

    if(voffset is None and 'v_dop_fit' in lineflux_data.columns):
        voffset=np.median(lineflux_data['v_dop_fit'])    #If Doppler shift is not specified, use median from lineflux_data if it exists
    if(voffset is None and not('v_dop_fit' in lineflux_data.columns)):
        voffset=0    #If Doppler shift is not defined, use 0 if lineflux_data has no element v_dop_fit
    w0*=(1+voffset*1e3/c.value)    #Apply Doppler shift

    #Make interpolation grid
    nvels=151
    nlines=np.size(w0)
    interpvel=np.arange(nvels)*dv-75.*dv
    interpind=np.zeros((nvels,nlines))+1  #keeps track of weighting for each velocity bin
    interpflux=np.zeros((nvels,nlines))
    interpuflux=np.zeros((nvels,nlines))

    #Loop through all w0 values
    for i,my_w0 in enumerate(w0):
        mywave = wave[(wave > (my_w0-0.003)) & (wave < (my_w0+0.003))]  #Find nearby wavelengths
        myflux = flux[(wave > (my_w0-0.003)) & (wave < (my_w0+0.003))]  #Find nearby fluxes
        myuflux = uflux[(wave > (my_w0-0.003)) & (wave < (my_w0+0.003))]  #Find nearby ufluxes
        myvel = c.value*1e-3*(mywave - my_w0)/my_w0                     #Convert wavelength to velocity
        try:
            f1=interp1d(myvel, myflux, kind='linear', bounds_error=False)   #Interpolate onto velocity grid
            interpflux[:,i]=f1(interpvel)
            f2 = interp1d(myvel, myuflux, kind='linear', bounds_error=False)
            interpuflux[:,i]=f2(interpvel)
            w=np.where((interpvel > np.max(myvel)) | (interpvel < np.min(myvel)) | (np.isfinite(interpflux[:,i]) != 1 )  ) #remove fluxes beyond edges, NaNs
            if(np.size(w) > 0):
                interpind[w,i]=0
                interpflux[w,i]=0
                interpuflux[w,i]=0
        except:
            pass
    numer=np.nansum(interpflux,1)
    denom=np.nansum(interpind,1)
    mybool=(denom==0)   #Find regions where there is no data
    numer[mybool]='NaN' #Set to NaN
    denom[mybool]=1
    interpflux=numer/denom

    # Normalize the flux errors
    numer_err = np.sqrt(np.nansum(interpuflux**2., 1))
    interpuflux = numer_err / denom

    if(norm=='Maxmin'):  #Re-normalize if desired
        interpflux=(interpflux-np.nanmin(interpflux))/np.nanmax(interpflux-np.nanmin(interpflux))

    return (interpvel,interpflux,interpuflux)


def calc_model_flux(stackvel, stackflux, stackuflux, cont_jy, tab):
    '''
    Parameters
    ---------
    stackvel: stacked line velocity list
    stackflux: stacked line flux list
    stackuflux: stacked line flux uncertainty list
    cont_jy: continuum flux in Jy
    tab: astropy table of wavelengths of transitions

    Returns
    ---------
    calc_line_flux: stacked line integrated flux in W/m^2
    calc_line_uflux: stacked line integrated flux uncertainty in W/m^2
    stackvel: masked stacked line velocity list (within 10 * FWHM of stacked line)
    stackflux: masked stacked line flux list
    stackuflux: masked stacked line flux uncertainty list
    '''

    #PLOT STACKED LINE PROFILE INCLUDING ERRORS
    plt.plot(stackvel, stackflux)
    plt.fill_between(stackvel, stackflux-stackuflux, stackflux+stackuflux, alpha=0.2)
    plt.ylabel('Normalized flux')
    plt.xlabel('Velocity [km/s]')

    #CALCULATE 10 * FWHM OF STACKED LINE PROFILE ASSUMING A GAUSSIAN
    popt, pcov = curve_fit(gaussian, stackvel[~np.isnan(stackflux)], stackflux[~np.isnan(stackflux)], sigma=stackuflux[~np.isnan(stackflux)], bounds=[[-np.inf, -np.inf, 0., -np.inf], [np.inf, np.inf, np.inf, np.inf]])
    fwhm10 = 10 * popt[2]

    #OVERLAY LOCATION OF 10*FWHM ON STACKED LINE PLOT
    plt.vlines(-fwhm10, ymin = 1., ymax=np.nanmax(stackflux))
    plt.vlines(fwhm10, ymin = 1., ymax=np.nanmax(stackflux))

    #FILTERS OUT VELOCITIES OUTSIDE THE 10 * FWHM RANGE
    fin_indices = np.where((stackvel > -fwhm10) & (stackvel < fwhm10))[0]
    stackvel = stackvel[fin_indices]
    stackflux = stackflux[fin_indices]
    stackuflux = stackuflux[fin_indices]

    #CALCULATES STACKED *LINE* FLUX BY FLUX NORMALIZING AND CONTINUUM SUBTRACTING
    stacklineflux = (stackflux*cont_jy)-cont_jy
    stacklineuflux = stackuflux*cont_jy

    #CALCULATES AVERAGE VELOCITY SEPARATION BETWEEN POINTS
    dvel=np.nanmean(np.diff(stackvel))

    #CALCULATES AVERAGE WAVELENGTH SEPARATION BETWEEN POINTS FOR EACH TRANSITION
    listofwavebins = [((dvel*1e9) / (c.value*1e6) * my_w0) for my_w0 in tab['wave']]
    dwave=np.abs(np.nanmean(listofwavebins))

    finflux = []
    finuflux = []

    #ITERATES THROUGH EACH WAVELENGTH OF EACH CO LINE
    for wave1 in tab['wave']:

        #CALCULATES FREQUENCY OF LINE
        nufit=c.value/(wave1*1e-6)

        #CONVERTS FLUX IN Jy*micron/s TO W/m^2
        conversion_factor = 1e-26*1e-6*nufit**2./c.value*un.W/un.m/un.m

        #CALCULATE STACKED LINE FLUX ASSUMING A CENTRAL WAVELENGTH GIVEN BY TABLE
        lineflux = np.nansum(stacklineflux)*dwave*conversion_factor
        finflux.append(lineflux)

        #CALCULATE STACKED LINE FLUX UNCERTAINTY ASSUMING A CENTRAL WAVELENGTH GIVEN BY TABLE
        lineuflux = np.sqrt(np.nansum(stacklineuflux**2.))*dwave*conversion_factor
        finuflux.append(lineuflux)

    #GRABS VALUES ONLY, EXCLUDING UNITS
    finfluxlist = [i.value for i in finflux]
    finufluxlist = [i.value for i in finuflux]

    #CALCULATE MEDIAN FLUX AND UNCERTAINTY, ASSUMING COMPOSITE PROFILE IS LOCATED AT EACH TRANSITION
    calc_line_flux = np.nanmedian(finfluxlist)
    calc_line_uflux = np.nanmedian(finufluxlist)

    return calc_line_flux, calc_line_uflux, stackvel, stacklineflux, stacklineuflux

def trans_wave_finder(wl_list, opt):
    '''
    Parameters
    ---------
    wl_list: list of identified transition wavelengths
    opt: selection of high- or low-J line regions
        allowed inputs are "high" and "low"

    Returns
    ---------
    wl_list: list of identified transition wavelengths for either high- or low-J lines
    '''

    #CREATES MASK FOR THIRD ORDER
    m1 = np.array(wl_list) < 4.9
    m2 = np.array(wl_list) > 4.64
    m = m1 & m2

    #MASKS HIGH-J AND LOW-J LINES
    wl_list1 = np.array(wl_list)[m]
    wl_list2 = np.array(wl_list)[np.array(wl_list) > 4.9]

    #RETURNS HIGH- OR LOW-J LINES ONLY DEPENDING ON USER INPUT
    if opt == 'low':
        wl_list = wl_list1
    if opt == 'high':
        wl_list = wl_list2

    return wl_list





def indivline_calc_and_fit_only_find_empty_vspace(wl, flux, flux_error, stackvel, stacklineflux, stacklineuflux, calc_line_flux, calc_line_uflux, cont_jy, wl_list, trans_list, hitran_data):
    '''
    Copied and pasted first several lines of indivline_calc_and_fit() to only find problem wavelengths and transitions. 
    
    Parameters
    ---------
    wl: data wavelength array
    flux: data flux array
    flux_error: data flux error array
    stackvel: stacked velocity array
    stacklineflux: flux normalized continuum subtracted stacked flux array
    stacklineuflux: flux normalized stacked flux uncertainty array
    calc_line_flux: integrated stacked line flux value
    calc_line_uflux: integrated stacked line flux uncertainty value
    cont_jy: continuum flux in Jy
    wl_list: list of identified, filtered (high- vs low-J) transition wavelengths
    trans_list: list of identified, filtered transition names
    hitran_data: HITRAN table with CO line properties

    Returns
    ---------
    tab: Astropy table with flux, flux uncertainty, and wavelength for all transitions
    '''

    # flag for whether to plot x-axis in wavelength or velocity
    plot_velocity = False

    # wavelength range over which to look for lines
    wave_range = (4.6,5.2)

    # wavelength window around the line to look for data (roughly corresponds to +/-30 km/s)
    delta_wave = 5e-4

    #DEFINE VARIABLES FROM INPUTS TO MESH WITH LATER CODE
    hitran_CO = hitran_data
    wave = wl
    uflux = flux_error
    trans_ids = trans_list
    trans_wave = wl_list

    # take snippets of each transition from +/-delta_v in km/s, sampled at 1 km/s
    # ~10% wider in wavelength to avoid interpolation issues
    delta_v = 200 #change depending on desired linewidth
    clight = 2.99792458e5
    delta_wave = 1.1 * 5 * delta_v / clight
    if plot_velocity:
        v_spec = np.arange(-delta_v, 1.01*delta_v, 1)

    #INITIALIZE PLOTTING PARAMETERS
    nrows = 5
    ncols = math.ceil(len(trans_wave) / nrows)


    calculated_linefluxes = []
    calculated_lineerrors = []
    problem_wavelengths = []
    problem_transitions = []
    
    #ITERATE THROUGH EACH TRANSITION
    for i, wave1 in enumerate(trans_wave):

        #GRAB TRANSITION WAVELENGTH
        wave1 = wave1
        col = i % ncols
        row = int(i / ncols)

        #FIND INDICES OF WAVELENGTH VALUES WITHIN delta_wave OF LINE CENTER
        j = np.where((wave > wave1-delta_wave) & (wave < wave1+delta_wave))[0]

        #RETRIEVE FILTERED WAVELENGTH VALUES
        wave_plot = wave[j]

        #CONVERT FILTERED WAVELENGTH VALUES TO VELOCITIES
        v_wave = (wave[j] / wave1 - 1) * clight

        dwave=np.nanmean(np.diff(wave_plot)) #WIDTH OF SINGLE PIXEL (BIN) IN MICRONS

        # normalize the continuum around each transition to unity
        # note that this just gives NaN if there isn't enough baseline either side of the line
        nj = j.size
        line_mask = np.ones(nj, dtype=bool)
        line_mask[int(nj/2-nj/4):int(nj+nj/4)] = False
        norm_flux = flux[j] / np.nanmedian(flux[j[line_mask]])
        norm_err = uflux[j] / np.nanmedian(flux[j][line_mask])

        #FOR PURPOSES OF FLUX CALCULATION, CONVERT AGAIN TO ACTUAL FLUX VALUE OF OBJ
        flux_jy = (norm_flux * cont_jy) - cont_jy
        uflux_jy = (norm_err * cont_jy)

        #INTERPOLATE ONTO SAME GRID HERE AS STACK ABOVE
        interp = interp1d(v_wave, flux_jy, fill_value = 'extrapolate')
        flux_velspace = interp(stackvel)
        interp_uflux = interp1d(v_wave, uflux_jy, fill_value = 'extrapolate')
        uflux_velspace = interp_uflux(stackvel)

        #FIND NAN INDICES FROM BOTH FLUX FROM DATA AND FLUX FROM MODEL AND REMOVE BOTH SETS
        mask1 = ~np.isnan(flux_velspace)
        mask2 = ~np.isnan(stacklineflux)
        mask = mask1 & mask2
        flux_velspace = flux_velspace[mask]
        stacklineflux_copy = stacklineflux[mask]
        stacklineuflux_copy = stacklineuflux[mask]

        #FIND APPROPRIATE SCALING TO MODEL USING CURVE_FIT
        if len(flux_velspace) == 0:
            print('!! Empty flux_vspace at transition at', wave1, trans_ids[i], 'in indivline_calc_and_fit(); need to from line list.')
            problem_wavelengths.append(wave1)
            problem_transitions.append(trans_ids[i])
            
    return problem_wavelengths, problem_transitions



def indivline_calc_and_fit(wl, flux, flux_error, stackvel, stacklineflux, stacklineuflux, calc_line_flux, calc_line_uflux, cont_jy, wl_list, trans_list, hitran_data):
    '''
    Parameters
    ---------
    wl: data wavelength array
    flux: data flux array
    flux_error: data flux error array
    stackvel: stacked velocity array
    stacklineflux: flux normalized continuum subtracted stacked flux array
    stacklineuflux: flux normalized stacked flux uncertainty array
    calc_line_flux: integrated stacked line flux value
    calc_line_uflux: integrated stacked line flux uncertainty value
    cont_jy: continuum flux in Jy
    wl_list: list of identified, filtered (high- vs low-J) transition wavelengths
    trans_list: list of identified, filtered transition names
    hitran_data: HITRAN table with CO line properties

    Returns
    ---------
    tab: Astropy table with flux, flux uncertainty, and wavelength for all transitions
    '''

    # flag for whether to plot x-axis in wavelength or velocity
    plot_velocity = False

    # wavelength range over which to look for lines
    wave_range = (4.6,5.2)

    # wavelength window around the line to look for data (roughly corresponds to +/-30 km/s)
    delta_wave = 5e-4

    #DEFINE VARIABLES FROM INPUTS TO MESH WITH LATER CODE
    hitran_CO = hitran_data
    wave = wl
    uflux = flux_error
    trans_ids = trans_list
    trans_wave = wl_list

    # take snippets of each transition from +/-delta_v in km/s, sampled at 1 km/s
    # ~10% wider in wavelength to avoid interpolation issues
    delta_v = 200 #change depending on desired linewidth
    clight = 2.99792458e5
    delta_wave = 1.1 * 5 * delta_v / clight
    if plot_velocity:
        v_spec = np.arange(-delta_v, 1.01*delta_v, 1)

    #INITIALIZE PLOTTING PARAMETERS
    nrows = 5
    ncols = math.ceil(len(trans_wave) / nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15), sharey=True)

    calculated_linefluxes = []
    calculated_lineerrors = []

    #ITERATE THROUGH EACH TRANSITION
    for i, wave1 in enumerate(trans_wave):

        #GRAB TRANSITION WAVELENGTH
        wave1 = wave1
        col = i % ncols
        row = int(i / ncols)

        #FIND INDICES OF WAVELENGTH VALUES WITHIN delta_wave OF LINE CENTER
        j = np.where((wave > wave1-delta_wave) & (wave < wave1+delta_wave))[0]

        #RETRIEVE FILTERED WAVELENGTH VALUES
        wave_plot = wave[j]

        #CONVERT FILTERED WAVELENGTH VALUES TO VELOCITIES
        v_wave = (wave[j] / wave1 - 1) * clight

        dwave=np.nanmean(np.diff(wave_plot)) #WIDTH OF SINGLE PIXEL (BIN) IN MICRONS

        # normalize the continuum around each transition to unity
        # note that this just gives NaN if there isn't enough baseline either side of the line
        nj = j.size
        line_mask = np.ones(nj, dtype=bool)
        line_mask[int(nj/2-nj/4):int(nj+nj/4)] = False
        norm_flux = flux[j] / np.nanmedian(flux[j[line_mask]])
        norm_err = uflux[j] / np.nanmedian(flux[j][line_mask])

        #FOR PURPOSES OF FLUX CALCULATION, CONVERT AGAIN TO ACTUAL FLUX VALUE OF OBJ
        flux_jy = (norm_flux * cont_jy) - cont_jy
        uflux_jy = (norm_err * cont_jy)

        #INTERPOLATE ONTO SAME GRID HERE AS STACK ABOVE
        interp = interp1d(v_wave, flux_jy, fill_value = 'extrapolate')
        flux_velspace = interp(stackvel)
        interp_uflux = interp1d(v_wave, uflux_jy, fill_value = 'extrapolate')
        uflux_velspace = interp_uflux(stackvel)

        #FIND NAN INDICES FROM BOTH FLUX FROM DATA AND FLUX FROM MODEL AND REMOVE BOTH SETS
        mask1 = ~np.isnan(flux_velspace)
        mask2 = ~np.isnan(stacklineflux)
        mask = mask1 & mask2
        flux_velspace = flux_velspace[mask]
        stacklineflux_copy = stacklineflux[mask]
        stacklineuflux_copy = stacklineuflux[mask]

        #FIND APPROPRIATE SCALING TO MODEL USING CURVE_FIT
        if len(flux_velspace) == 0:
            print('!! Empty flux_vspace at transition at', wave1, trans_ids[i], 'in indivline_calc_and_fit(); deleting from line list...')
            #problem_wavelength = wave1
            #problem_transition = trans_ids[i]
            #r_list.append((round(problem_wavelength,7), problem_transition)) # skipping for now, should probably do this
            #print(np.where(np.array(wl_list) == problem_wavelength)[0][0], np.where(np.array(trans_list) == problem_transition)[0][0])
            #wl_list.pop(np.where(np.array(wl_list) == problem_wavelength)[0][0])
            #trans_list.pop(np.where(np.array(trans_list) == problem_transition)[0][0])
            #continue
            
        popt, pcov = curve_fit(model, stacklineflux_copy, flux_velspace, sigma=stacklineuflux_copy, p0=[1.], bounds = [0., 2.])
        #ERRORS ARE ON ORDER OF 5-10%

        #SCALE INDIV LINE FLUX BY POPT
        modelflux = calc_line_flux * popt
        modeluflux = modelflux * (np.sqrt((calc_line_uflux / calc_line_flux)**2.+(np.sqrt(np.diag(pcov)) / popt)**2.))
        stackwave = ((stackvel*1e9) / (c.value*1e6) * wave1) + wave1 #wavelength region for each line for plotting

        calculated_linefluxes.append(modelflux)
        calculated_lineerrors.append(modeluflux)

        if plot_velocity:
            pass
    #         flux_v = interp1d(v_wave, norm_flux, fill_value='extrapolate')
    #         flux_err = interp1d(v_wave, norm_err, fill_value='extrapolate')
    #         f_spec = flux_v(v_spec)
    #         axs[row, col].plot(v_spec, f_spec)
    #         axs[row, col].set_xlim(-delta_v, delta_v)
    #         axs[row, col].axvline(0, color='C1', ls = '--')
    #         axs[row, col].text(3, 0.7, trans_ids[i], rotation=90)

        else:
            axs[row, col].plot(wave_plot, flux_jy, label = 'Data')
            axs[row, col].fill_between(wave_plot, flux_jy - uflux_jy, flux_jy + uflux_jy, alpha = 0.3)
            axs[row, col].plot(stackwave, stacklineflux * popt, label = 'Stacked model')
            axs[row, col].set_xlim(wave1-delta_wave, wave1+delta_wave)
            axs[row, col].axvline(wave1, color='C1', ls = '--')
            axs[row, col].annotate(trans_ids[i][0], (0.1, 0.8), xycoords='axes fraction')
            axs[row, col].annotate('{:0.3e}'.format(modelflux[0]) + '+/-' + '{:0.3e}'.format(modeluflux[0]), (0.1, 0.72), xycoords='axes fraction')
            #axs[row, col].legend()

        #axs[row, col].set_ylim(0.2, 1.5)

        if plot_velocity:
            axs[row, col].set_xlabel(r'Velocity (km/s)')
        else:
            axs[row, col].set_xlabel(r'$\lambda$ ($\mu$m)')
        if col == 0:
            axs[row, col].set_ylabel('Continuum subtracted flux [Jy]')

    # don't show axes for panels where there is no data
    n_extra = nrows * ncols - len(trans_wave)
    if n_extra > 0:
        for i in range(n_extra):
            col = (nrows*ncols - i - 1) % ncols
            row = int((nrows*ncols - i - 1) / ncols)
            axs[row, col].axis('off')

    #CREATE TABLE WITH TRANSITION WAVELENGTH, FLUX AND UFLUX
    tab = Table([trans_wave * u.micron, calculated_linefluxes * u.W / u.m / u.m, calculated_lineerrors * u.W / u.m / u.m], names=(['wave', 'flux', 'uflux']))

    return tab
