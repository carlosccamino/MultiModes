#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodes2 

MultiModes is a highly customisable code to extract the most significant 
frequencies from a set of time series. For each of them, it calculates
the LombScargle periodogram and performs a nonlinear simultaneous fit, 
using a multisine function, to a set number of frequencies. As a stop 
criterion, you can choose between the FAP or the SNR criterion,
depending on the type of analysis you want to perform.

Author: Javier Pascual Granado (IAA-CSIC)
- Adapted from the original code from David Pamos Ortega (UGR).

Contributors: Antonio García, Sebastiano de Franciscis, Cristian Rodrigo

Version: 0.1.4 (see CHANGES)
16:10 feb 20, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from astropy.timeseries import LombScargle
from astropy.io import fits
from lmfit import Parameters, minimize
from scipy import stats
from scipy.optimize import curve_fit
import os
import argparse
import glob
from timeit import default_timer as timer
import nifty_ls


# ---- GLOBAL VARIABLES ----

""" boxR is the number of bins to use for noise estimation
Reasonable values for boxR range from 4 to 40, depending on the duration of the time series
and the signal under study.
A typical mode density in dSct is 5 peaks per c/d.
For a TESS sector (27 d), 2 c/d (typical value in literature) corresponds to 54 bins. 
"""
boxR = 40

# ---- FUNCTION DEFINITIONS ----

def sinusoid(time, A, fre, ph, B):
        """Sine function to fit a harmonic component with adjusted freq."""
        # (Aug 13, 2025) added B constant for floating mean, should avoid subtracting mean every iteration
        sinu = A*np.sin(2*np.pi*(fre*time + ph)) + B
        return sinu

def initial_phase(x, y, best_freq, hifac=1, ofac=4.0):
    """ This is my 'phispec.phases' function modified in order to output 
     just the phase of the maximum in LS periodogram."""
    n = len(x)
    nout = int(0.5 * ofac * hifac * n)
    nmax = nout + 1
    twopi = 2*np.pi
    ave = np.mean(y)

    xmax = np.max(x)
    xmin = np.min(x)
    xdif = xmax - xmin
    xave = 0.5 * (xmax + xmin)
    pnow = 1.0 / (xdif * ofac)
    arg = twopi * ((x - xave) * pnow)
    wpr = -2.0 * np.sin(0.5 * arg) ** 2
    wpi = np.sin(arg)
    wr0 = np.cos(arg)
    wi0 = wpi

    yy = (y - ave)
    wk1 = pnow*(np.arange(nmax) + 1)
    ibest = np.abs(wk1 - best_freq).argmin()

    # Compute alpha in complex form
    alpha = (1 + wpr) + 1j * wpi
    r = np.abs(alpha)
    theta = np.angle(alpha)

    # Compute alpha^i using polar form
    alpha = (r ** ibest) * (np.cos(ibest * theta) + 1j * np.sin(ibest * theta))

    # Compute wr and wi at iteration i
    wr = (alpha * (wr0 + 1j * wi0)).real
    wi = (alpha * (wr0 + 1j * wi0)).imag

    # Now use wr and wi in the original computation
    sumsh = np.sum(wr * wi)
    sumc = np.sum((wr - wi) * (wr + wi))
    wtau = 0.5 * np.arctan2(2.0 * sumsh, sumc)
    arg0 = -twopi * xave * wk1[ibest] - wtau
    swtau = np.sin(wtau)
    cwtau = np.cos(wtau)
    ss = wi * cwtau - wr * swtau
    cc = wr * cwtau + wi * swtau
    sums = np.sum(ss ** 2)
    sumc = np.sum(cc ** 2)
    sumsy = np.sum(yy * ss)
    sumcy = np.sum(yy * cc)

    ph = (-np.arctan2(sumsy / np.sqrt(sums), sumcy / np.sqrt(sumc)) + arg0 ) % twopi
    ph = twopi - ph
    ph = ph/twopi

    return ph

def hf_noise_estimate(time, flux, osratio, n=100):
    """Estimation of noise using the mean of near-Nyquist frequency bins."""
    L = len(time)
    tt = time[-1]-time[0]
    avnyq = 0.5*L/tt
    fre_step = 1/tt/osratio
    if L <= n:
        n = L-1

    # Indexes and freqs
    rind = int(avnyq/fre_step) - 1
    lind = rind - n
    fre = fre_step*np.arange(lind, rind)
    nifty_res = nifty_ls.lombscargle(time, flux, normalization='psd', center_data=True, fit_mean=False, 
                             fmin=fre[0], fmax=fre[-1], Nf=len(fre))
    amps = 2*np.sqrt(nifty_res.power/L)

    return np.mean(amps)

def global_noise_estimate(time, flux, osratio):
    """Estimation of noise using the median of the whole periodogram."""
    L = len(time)
    ls = LombScargle(time, flux, normalization='psd',
                     center_data=True, fit_mean=False)
    # _, power = ls.autopower(samples_per_peak=osratio)
    _, power = ls.autopower(samples_per_peak=osratio, method="fastnifty")
    amps = 2*np.sqrt(power/L)

    return np.median(amps)

def box_noise_estimate(time, flux, best_freq, ampmax, indmpow, osr=10):
    """Estimation of noise using a box around the frequency in the periodogram."""
    # (21 Aug, 2025) introduced a factor of 4 to guarantee well resolved peaks
    # in the noise estimation.
    osratio = osr*4

    # box in units of the Rayleigh freq. [nu - box/2, nu + box/2]
    box = boxR*osratio
    N = len(time)

    # Initial guess for the phase
    best_ph = initial_phase(time, flux, best_freq, ofac=osratio)

    # Least squares fitting
    # (14 Aug, 2025) bounds added to curve_fit to suppress negative amplitudes
    dela = ampmax/2
    minamp = ampmax - dela
    maxamp = ampmax + dela
    tt = time[-1]-time[0]
    R = 1/tt
    delf = R/2
    minfreq = best_freq - delf
    maxfreq = best_freq + delf
    initial_guess = np.array([ampmax, best_freq, best_ph, 0.0])
    # (Aug 17, 2025) interval is extended to allow more flexibility
    minph = 0
    maxph = 1
    bounds = ([minamp, minfreq, minph, -np.inf], [maxamp, maxfreq, maxph, np.inf])
    params,_ = curve_fit(sinusoid, time, flux, p0=initial_guess, bounds=bounds, max_nfev=100*4*20)
    y_fit = sinusoid(time, *params)
    res_fit = y_fit - flux

    # Box indexes and freqs
    lind = indmpow - int(box/2)
    rind = indmpow + int(box/2) + 1
    fre_step = best_freq/indmpow
    fre = fre_step*np.arange(lind, rind)

    # Periodogram of the residuals
    nif = nifty_ls.lombscargle(time, res_fit, normalization='psd', center_data=True, fit_mean=False, 
                                 fmin=fre[0], fmax=fre[-1], Nf=len(fre))
    res_amps = 2*np.sqrt(nif.power/N)
    noise = np.median(res_amps)
    snr = params[0]/noise
        
    return noise, snr

def noise_spectrum(time, flux, osratio, kernel_size):
    """Estimation of noise spectrum using a median filter on the amplitude spectrum."""
    ls = LombScargle(time, flux, normalization='psd',
                     center_data=True, fit_mean=False)
    _, power = ls.autopower(samples_per_peak=osratio)
    L = len(time)
    amps = 2*np.sqrt(power/L)

    if kernel_size % 2 == 0: # fix for even kernel size
        kernel_size = kernel_size + 1
    
    pad_size = kernel_size // 2
    padded_amps = np.pad(amps, pad_size, mode='edge')
    filtered_amps = np.zeros_like(amps)

    for i in range(len(amps)):
        # Extract the window
        window = padded_amps[i:i + kernel_size]
        # Compute the median
        filtered_amps[i] = np.median(window)

    return filtered_amps

def periodogram(time, flux, osratio, max_freq): 
    '''Fast Lomb Scargle to calculate the periodogram (Press & Ribicky 1989)'''
    ls = LombScargle(time, flux, normalization = 'psd', 
                     center_data = True, fit_mean = False)
    frequency, power = ls.autopower(maximum_frequency = max_freq, 
                                    samples_per_peak = osratio, 
                                    method="fastnifty")
    indmpow = np.argmax(power)
    best_frequency = frequency[indmpow]
    amps = 2*np.sqrt(power/len(time))
    ampmax = amps[indmpow]

    return ls, frequency, amps, best_frequency, ampmax, power, indmpow

def fit(t, params):
    '''Multi-sine fit function with all the parameters of frequencies, amplitudes, phases'''
    y = 0
    pars = params.valuesdict()
    amps_dict = {k:pars[k] for k in pars if 'a' in k}
    freqs_dict = {k:pars[k] for k in pars if 'b' in k}
    phs_dict = {k:pars[k] for k in pars if 'c' in k}
    # (Aug 13, 2025) added b values as floating mean correction for the fitting
    freqs = list([pars[k] for k in freqs_dict])
    amps = list([pars[k] for k in amps_dict])
    phs = list([pars[k] for k in phs_dict])
    b = [pars[k] for k in pars if 'm' in k]
    for (a, f, p) in zip(amps, freqs, phs):
        y += sinusoid(t, a, f, p, b) 
    return y, amps, freqs, phs

def residual(params, t, flux): 
    '''Residual between the model and data'''
    res = fit(t, params)[0] - flux

    return res

def lightcurve(file, isfits, isascii, timecol, fluxcol, header_lines): 
    '''Reading the file to extract all data'''
    if isfits:
        hdul = fits.open(file)
        data = hdul[1].data
        data = pd.DataFrame(np.array(data))
        time = np.array( data.iloc[:,timecol-1], dtype=np.float64 )     # extract times
        fluxes = np.array( data.iloc[:,fluxcol-1], dtype=np.float64 )   # extract fluxes
    elif isascii:
        data = np.loadtxt(file, skiprows=header_lines)
        time = data[:,timecol-1]     # extract times
        fluxes = data[:,fluxcol-1]  # extract fluxes

    time = time - time[0]
    T = time[-1] - time[0]
    N = len(time)
    r = 1/T   # Rayleigh frequency resolution

    return time, fluxes, T, N, r


def snr_or_fap(par, min_snr = np.nan, S_N = np.nan, max_fap = np.nan, all_faps = np.nan): 
    '''Choosing between the SNR or the FAP stop criterion. 
       SNR by default'''
    if 'SNR' in par:
        return min_snr, S_N
    elif 'FAP' in par:
        return max_fap,  all_faps


"""
def comb_freqs(pd):
    ''' Function to detect the combination frequencies once all of them having
    been extracted'''

    freqs = pd['Freqs']
    amps = pd['Amps']
    f1 = freqs[0]
    f2 = freqs[1]
    f_lim = 100.

    n1_list = np.arange(-2, 2).tolist()
    n2_list =np.arange(-2, 2).tolist()
    children1_2 = []

    n1_n2 = []


    for i in n1_list:
        for j in n2_list:
            candidate = i*f1 + j*f2
            if 0 < candidate < f_lim and candidate != f1 and candidate !=f2:
                children1_2.append(round(i*f1 + j*f2, 2))
                n1_n2.append((i,j,0))



    comb_freqs = []

    comb_amps = []

    n1_n2_values = []

    error = R

    for (ch, v) in zip(children1_2, n1_n2):
        for (f,a) in zip(freqs,amps):
            if abs(ch-f)<error:
                comb_freqs.append(f)
                comb_amps.append(a)
                n1_n2_values.append(v)

    return comb_freqs, comb_amps, n1_n2_values
"""

def arguments(agrv):
    '''Function to parse the command line arguments'''
    parser = argparse.ArgumentParser(description='Open the directory with the \
                                    files to be processed')
    parser.add_argument('--p', type=str, help='Select a parameters file')
    parser.add_argument('--d', type=str, help='Select a directory \
                            containing a list of files')
    parser.add_argument('--file', type=str, help='Select a single \
                            file to be opened')
    args = parser.parse_args(agrv)

    return args

# Here begins the main part of the code
def multimodes(args, dash = 100*'-'): 

    # Initializing the necessary lists

    all_faps = []  # FAP values for each extracted frequency
    S_N = []       # SNR values for each extracted frequency

    # Reading the initial file with the values of the parameters, if it exists
    if args.p:
        paramname = args.p
    else:
        paramname = 'ini.txt'
    

    # ---- Default initial parameters
        
    osratio = 5  # Oversampling ratio
    max_freq = 100  # Max frequency in the periodogram
    sim_fit_n = 20  # Max number of frequencies of the simultaneous fit
    stop = 'SNR'  # Stop criterion # Multimodes
    min_snr = 4.0  # Min value of Signal-to-Noise Ratio (SNR) to detect a frequency
    max_fap = 0.01  # Max value of False Alarm Probability (FAP)
    timecol = 1  # column order for time and fluxes
    fluxcol = 2
    #save_plot_per = 0  # save plots of periodogram every xx iterations
    save_data_res = 0  # save data of residual every xx iterations
    save_freq_list = 0 # save result files (freq.list) every xx iterations (must be a multiple of sim_fit_n)
    save_plot_resps = 0  # write residual spectrum after last iteration
    max_iter = 1000 # Multimodes
    header_lines = 1 # skip header lines # lightcurve
    clean_close = 0 # clean frequencies that are closer than Rayleigh 
    noise_method = 'hf' # method to estimate the noise level

    if os.path.isfile(paramname):
        with open(paramname, 'r') as file:
            for line in file:
                line.replace("\n", "")
                if line.startswith("sim_fit_n"):
                    sim_fit_n = int(line.split(' ')[1])
                if line.startswith("osratio"):
                    osratio = float(line.split(' ')[1])
                if line.startswith("max_freq"):
                    max_freq = float(line.split(' ')[1])
                if line.startswith("max_fap"):
                    max_fap = float(line.split(' ')[1])
                if line.startswith("min_snr"):
                    min_snr = float(line.split(' ')[1])
                if line.startswith("stop"):
                    stop = str(line.split(' ')[1])
                if line.startswith("timecol"):
                    timecol = int(line.split(' ')[1])
                if line.startswith("fluxcol"):
                    fluxcol = int(line.split(' ')[1])
                if line.startswith("save_data_res"):
                    save_data_res = int(line.split(' ')[1])
                if line.startswith("save_plot_resps"):
                    save_plot_resps = int(line.split(' ')[1])
                if line.startswith("header_lines"):
                    header_lines = int(line.split(' ')[1])
                if line.startswith("max_iter"):
                    max_iter = int(line.split(' ')[1])
                    if max_iter==0:
                        max_iter = 1e6
                if line.startswith("save_freq_list"):
                    save_freq_list = int(line.split(' ')[1])
                if line.startswith("clean_close"):
                    clean_close = int(line.split(' ')[1])
                if line.startswith("noise_method"):
                    noise_method = line.split(' ')[1]
    else:
        print('No param file. Default values will be used:')


    print('Number of frequencies of the simultaneous fit: ' + str(sim_fit_n))
    print('Samples per peak: ' + str(osratio))
    print('Maximum frequency: ' + str(max_freq))
    if stop == 'SNR':
        print('Stop Criterion: SNR > ' + str(min_snr))
    elif stop == 'FAP':
        print('Stop Criterion: FAP < ' + str(max_fap))


    # Creating the list with all the files
    isfits = False
    isascii = False
    if args.d:
        pth = './'+args.d+'/'
        fits_files = [f for f in glob.glob(pth+'*.fits')]
        ascii_files = [f for f in glob.glob(pth+'*.dat')]
        if len(ascii_files) == 0:
            ascii_files = [f for f in glob.glob(pth+'*.txt')]
            
        if len(ascii_files) == 0:
            if len(fits_files) == 0:
                print("No input files found.")

            fname = [os.path.splitext(f)[0] for f in fits_files]
            fits_files = sorted(fits_files)
            filepath = fits_files
            isfits = True
        else:
            fname = [os.path.splitext(f)[0] for f in ascii_files]
            ascii_files = sorted(ascii_files)
            filepath = ascii_files
            isascii = True

        fname = sorted(fname)

    else:
        pth = './/'
        fname = list(os.path.splitext(args.file))
        if fname[1] == '.fits':
            fits_files = ['./'+args.file]
            filepath = fits_files
            isfits = True
        elif fname[1] == '.dat' or fname[1] == '.txt':
            ascii_files = ['./'+args.file]
            filepath = ascii_files
            isascii = True

        fname = [fname[0]]

    # Starting with the iterative analysis
    columns = ['Number',
            'f',
            'A',
            stop]

    # Setting stop criterion
    parade = snr_or_fap(stop, min_snr = min_snr, S_N = S_N)[0] # Now, we have to pass the values of min_snr and S_N
    snr_or_faps = snr_or_fap(stop, max_fap = max_fap, all_faps = all_faps)[1] # Now, we have to pass the values of max_fap and all_faps

    # Initialization of the parameter structure
    params = Parameters()

    for (f, nm) in zip(filepath, fname):
        start = timer()  # Counting executing time for every analysed light curve
        n = 1
        num = 1

        data = lightcurve(f, isascii=isascii, isfits=isfits, 
                          timecol=timecol, fluxcol=fluxcol, header_lines=header_lines)  # Extracting data of every light curve
        time = data[0]     # Time vector
        lc = data[1]       # Light curve
        T = data[2]        # Total time span
        N = data[3]        # Number of points
        R = data[4]        # Rayleigh resolution

        sigma_lc = stats.sem(list(lc))  # 1-sigma error of fluxes
        sigma_amp = np.sqrt(2/N)*sigma_lc  # 1-sigma error of the amplitude

        # Define the path for the output

        if os.path.isfile(paramname):
            if args.d:
                ofolder = './results/' + pth[2:-1] + '+' + paramname[:-4] + '/'
                filesp = nm[2:].split('/')
                file = filesp[1]
                newpath = ofolder + file + '/'
                
            else:
                ofolder = './results/' + '+' + paramname[:-4] + '/'
                newpath = ofolder + nm + '/' 
                
        else:
            newpath = './results/' + nm + '/'

        if not os.path.exists(newpath):
            os.makedirs(newpath)

        # Calculating the initial periodogram
        lc0 = lc - np.mean(lc)
        lc = lc0

        # Noise level estimation
        if noise_method == "hf":
            no = hf_noise_estimate(time, lc, osratio)            
        else: 
            if noise_method == "g":
                no = global_noise_estimate(time, lc, osratio)

        """ Initialization of the lists to save the extracted frequencies,
        amplitudes and phases"""
        all_best_freqs = []
        all_max_amps = []
        all_phs = []
        snr_or_faps = []
        all_sigma_amps = []
        all_sigma_freqs = []
        all_sigma_phs = []
        all_rms = []
        params = Parameters()

        # Print header table
        print(dash)
        print('{:<15s}{:>18s}{:>25s}{:>32}' .format(columns[0],
                                                    columns[1],
                                                    columns[2],
                                                    columns[3]))
        print(dash)

        while n <= sim_fit_n:
            ls = periodogram(time, lc, osratio=osratio, max_freq=max_freq)
            freq = ls[3]  # freq at max power
            amp = ls[4]   # amp at max power
            imax = ls[6]  # index of max power
            rms = np.sqrt(sum(lc**2)/N)

            if noise_method == "box":
                no, snr = box_noise_estimate(time, lc, freq, amp, imax, osr=osratio)
            else:
                snr = amp/no
                
            var = (N/4)*no**2
            fap = 1 - (1 - np.exp(-ls[5].max()/var))**(N/2)

            # Print table of extracted frequencies from the LS periodogram
            data = [num, freq, amp, snr]
            print('{:<15d}{:>18f}{:>25f}{:>32}'.format(data[0],
                                                    data[1],
                                                    data[2],
                                                    data[3]))

            if parade == min_snr:
                if snr > parade and num < max_iter:  # Stop criterion
                    snr_or_faps.append(snr)
                    all_rms.append(rms)
                    all_sigma_amps.append(sigma_amp)

                    # Non-linear least squares fitting
                    dela = amp/2
                    minamp = amp - dela
                    maxamp = amp + dela
                    delf = R/2
                    minfreq = freq - delf
                    maxfreq = freq + delf
                    ph = initial_phase(time, lc, freq, ofac=osratio)
                    # (Aug 17, 2025) interval is extended to allow more flexibility
                    minph = 0
                    maxph = 1
                    # The old values for the search were ph = 0.5, minph=0, maxph=1
                    params.add('p_'+str(n)+'a', value=amp, min=minamp, max=maxamp)
                    params.add('p_'+str(n)+'b', value=freq, min=minfreq, max=maxfreq)
                    params.add('p_'+str(n)+'c', value=ph, min=minph, max=maxph)
                    # (Aug 13, 2025) added p_m for the B parameter fitting the floating mean
                    if n==1:
                        params.add('p_m', value=0.0)

                    # best_freqs = fit(time, params)[2]
                    max_amps = fit(time, params)[1]
                    
                    # Residuals
                    res = minimize(residual, params, args=(time, lc0),
                                method='least_squares')
                    lc = res.residual
                    params = res.params

                    # Error estimation
                    sigma_freqs = [np.sqrt(6/N)/(pi*T)*sigma_lc/np.abs(a)
                                for a in max_amps]
                    sigma_phs = [sigma_amp/np.abs(a) for a in max_amps]
                    
                    # Save residuals
                    if save_data_res != 0:
                        if np.mod(num, save_data_res) == 0:
                            lc_df = pd.DataFrame({'Time':time, 'Flux':list(lc)})
                            lc_df.to_csv(newpath+'res_'+str(num)+'.dat', sep = ' ',
                                        index=False, header = None)
                    
                    n += 1
                    num += 1
                    if n > sim_fit_n:
                        all_best_freqs += fit(time, params)[2]
                        all_max_amps += fit(time, params)[1]
                        all_phs += fit(time, params)[3]
                        all_sigma_freqs += sigma_freqs
                        all_sigma_phs += sigma_phs
                        lc0 = lc - np.mean(lc)
                    
                        # Write the frequency list
                        if save_freq_list !=0:
                            if np.mod(num, save_freq_list) == 0:
                                prew_df = pd.DataFrame({'Freqs': all_best_freqs,
                                                        'Amps': all_max_amps,
                                                        'Phases': all_phs,
                                                        'Amplitude 1-sigma error (mmag)': all_sigma_amps,
                                                        'Frequency 1-sigma error (c/d)': all_sigma_freqs,
                                                        'Phase 1-sigma error (c/d)': all_sigma_phs,
                                                        'SNR/FAP': snr_or_faps,
                                                        'rms': all_rms}
                                                    )
                                prew_df.to_csv(newpath+'freqs_'+str(num-1)+'.dat', sep=' ',
                                                index=False, header=None)
                        
                        n = 1
                        params = Parameters()
                else:
                    if n != 1:
                        all_best_freqs += fit(time, params)[2]
                        all_max_amps += fit(time, params)[1]
                        all_phs += fit(time, params)[3]
                        all_sigma_freqs += sigma_freqs
                        all_sigma_phs += sigma_phs

                    break

            elif parade == max_fap:
                if fap < parade:  # Stop criterion
                    snr_or_faps.append(fap)
                    all_rms.append(rms)
                    all_sigma_amps.append(sigma_amp)

                    # Non-linear least squares fitting
                    dela = amp/2
                    minamp = amp - dela
                    maxamp = amp + dela
                    delf = R/2
                    minfreq = freq - delf
                    maxfreq = freq + delf
                    ph = initial_phase(time, lc, freq, ofac=osratio)
                    # (Aug 17, 2025) interval is extended to allow more flexibility
                    minph = 0
                    maxph = 1
                    # The old values for the search were ph = 0.5, minph=0, maxph=1
                    params.add('p_'+str(n)+'a', value=amp, min=minamp, max=maxamp)
                    params.add('p_'+str(n)+'b', value=freq, min=minfreq, max=maxfreq)
                    params.add('p_'+str(n)+'c', value=ph, min=minph, max=maxph)
                    # (Aug 13, 2025) added p_m for the B parameter fitting the floating mean
                    if n==1:
                        params.add('p_m', value=0.0)

                    # best_freqs = fit(time, params)[2]
                    max_amps = fit(time, params)[1]
                    
                    # Residuals
                    res = minimize(residual, params, args=(time, lc0),
                                method='least_squares')
                    lc = res.residual
                    params = res.params
                    
                    # Error estimation
                    sigma_freqs = [np.sqrt(6/N)/(pi*T)*sigma_lc/np.abs(a)
                                for a in max_amps]
                    sigma_phs = [sigma_amp/np.abs(a) for a in max_amps]
                    data = [num, freq, amp, fap]
                    print('{:<15d}{:>18f}{:>25f}{:>32}'.format(data[0],
                                                            data[1],
                                                            data[2],
                                                            data[3]))
                    
                    # Save residuals
                    if save_data_res != 0:
                        if np.mod(num, save_data_res) == 0:
                            lc_df = pd.DataFrame({'Time':time, 'Flux':list(lc)})
                            lc_df.to_csv(newpath+'res_'+str(num)+'.dat', sep = ' ',
                                        index=False, header = None)
                    
                    n += 1
                    num += 1
                    if n > sim_fit_n:
                        all_best_freqs += fit(time, params)[2]
                        all_max_amps += fit(time, params)[1]
                        all_phs += fit(time, params)[3]
                        all_sigma_freqs += sigma_freqs
                        all_sigma_phs += sigma_phs
                        lc0 = lc - np.mean(lc)

                        # Write the frequency list
                        if save_freq_list !=0:
                            if np.mod(num, save_freq_list) == 0:
                                prew_df = pd.DataFrame({'Freqs': all_best_freqs,
                                                        'Amps': all_max_amps,
                                                        'Phases': all_phs,
                                                        'Amplitude 1-sigma error (mmag)': all_sigma_amps,
                                                        'Frequency 1-sigma error (c/d)': all_sigma_freqs,
                                                        'Phase 1-sigma error (c/d)': all_sigma_phs,
                                                        'SNR/FAP': snr_or_faps,
                                                        'rms': all_rms}
                                                    )
                                prew_df.to_csv(newpath+'freqs_'+str(num-1)+'.dat', sep=' ',
                                                index=False, header=None)
                        
                        n = 1
                        params = Parameters()
                else:
                    if n != 1:
                        all_best_freqs += fit(time, params)[2]
                        all_max_amps += fit(time, params)[1]
                        all_phs += fit(time, params)[3]
                        all_sigma_freqs += sigma_freqs
                        all_sigma_phs += sigma_phs
                    break
        
        prew_df = pd.DataFrame({'Freqs': all_best_freqs,
                        'Amps': all_max_amps,
                        'Phases': all_phs,
                        'Amplitude 1-sigma error (mmag)': all_sigma_amps,
                        'Frequency 1-sigma error (c/d)': all_sigma_freqs,
                        'Phase 1-sigma error (c/d)': all_sigma_phs,
                        'SNR/FAP': snr_or_faps,
                        'rms': all_rms})

        # Filtering the frequencies that are closer than the Rayleigh resolution
        # It compares amplitudes and removes less significant frequencies from the 
        # list of filtered frequencies. This might be useful only when osratio>1
        if clean_close != 0:
            added_freqs = []
            added_amps = []
            copied_freqs = all_best_freqs.copy()
            copied_amps = all_max_amps.copy()
            filtered_freqs = set(all_best_freqs)  # faster membership testing

            for (freq, amp) in zip(all_best_freqs, all_max_amps):
                if freq in filtered_freqs:
                    # Membership checks
                    if freq in copied_freqs:
                        copied_freqs.remove(freq)
                    if amp in copied_amps:
                        copied_amps.remove(amp)

                    added_freqs.append(freq)
                    added_amps.append(amp)

                    for (f, a) in zip(copied_freqs, copied_amps):
                        if added_freqs and (added_freqs[-1] - R < f < added_freqs[-1] + R):
                            if added_amps[-1] > a:
                                if f in filtered_freqs:
                                    filtered_freqs.remove(f)
                                    prew_df = prew_df[prew_df.Freqs != f]
                            else:  # If the current amplitude is not greater, remove the current frequency
                                if freq in filtered_freqs:
                                    filtered_freqs.remove(freq)
                                    prew_df = prew_df[prew_df.Freqs != freq]
                else:
                    # If the frequency is not in filtered_freqs, remove it from copied lists
                    if freq in copied_freqs:
                        copied_freqs.remove(freq)
                    if amp in copied_amps:
                        copied_amps.remove(amp)


        # Filtering the linear combinations of frequencies
        """ TO REVISE AND UNCOMMENT AFTER TESTING
        try:
            combineds = comb_freqs(prew_df)[0]

            for f in combineds:
                prew_df = prew_df[prew_df.Freqs != f]
        except KeyError:
            pass
        """

        # Write residual lightcurve after last iteration
        reslc = pd.DataFrame({'Time': time, 'Residuals': lc})
        reslc.to_csv(newpath+'res_lc.dat', sep=' ', index=False, header=None)

        # Write the frequency list                            )
        prew_df.to_csv(newpath+'best_modes.dat', sep=' ', index=False,
                    header=None)

        # Save the initial periodogram as ASCII file and line plot
        """ ADD FLAG FOR THIS BEFORE UNCOMMENT
        per.to_csv(newpath+'pg.dat', sep=' ', index=False, header = None)
        per.plot(kind = 'line', x='Frequency', y='Amplitude', title = nm,
                legend = False)
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.savefig(newpath+'LS.png')
        plt.close()
        """

        # Write residual spectrum after last iteration
        if save_plot_resps !=0:
            resps = pd.DataFrame({'Frequencies': ls[1], 'Amplitudes': ls[2]})
            resps.to_csv(newpath+'res_ps.dat', sep=' ', index=False, header = None)

        end = timer()

        print('Executing Time: ' + str(end-start))

    return

#=========================================

if __name__ == "__main__":

    import sys

    # Multimodes starts
    dash = 100*'-'
    print(dash)
    print('Running MultiModes'.center(110))
    print(dash)

    # Reading the arguments
    args = arguments(agrv = sys.argv[1:])

    # Calling the primary function
    multimodes(args)

    # Finishing the execution
    print(dash)
    print('Finished'.center(110))
    print(dash)
