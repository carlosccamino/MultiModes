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

Contributors: Antonio García, Sebastiano de Franciscis, Cristian Rodrigo,
              Carlos C. Camino Mesa.

Version: 0.1.5 (resolution-constrained simultaneous fitting)
05 Oct 2026
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from astropy.timeseries import LombScargle
from astropy.io import fits
from lmfit import Parameters, minimize
from scipy.optimize import curve_fit
import argparse
import glob
from timeit import default_timer as timer
import nifty_ls
import warnings
warnings.filterwarnings(
    "ignore",
    message="Using UFloat objects with std_dev==0 may give unexpected results."
)


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

    # Avoid invalid phase estimates for degenerate or non-finite data.
    if (
        not np.isfinite(sums)
        or not np.isfinite(sumc)
        or sums <= 0
        or sumc <= 0
    ):
        return 0.0
    sumsy = np.sum(yy * ss)
    sumcy = np.sum(yy * cc)

    ph = (-np.arctan2(sumsy / np.sqrt(sums), sumcy / np.sqrt(sumc)) + arg0) % twopi
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

    nifty_res = nifty_ls.lombscargle(
        time,
        flux,
        normalization='psd',
        center_data=True,
        fit_mean=False,
        fmin=fre[0],
        fmax=fre[-1],
        Nf=len(fre)
    )

    amps = 2*np.sqrt(nifty_res.power/L)

    return np.mean(amps)


def global_noise_estimate(time, flux, osratio):
    """Estimation of noise using the median of the whole periodogram."""
    L = len(time)

    ls = LombScargle(
        time,
        flux,
        normalization='psd',
        center_data=True,
        fit_mean=False
    )

    _, power = ls.autopower(
        samples_per_peak=osratio,
        method="fastnifty"
    )

    amps = 2*np.sqrt(power/L)

    return np.median(amps)


def box_noise_estimate(time, flux, best_freq, ampmax, indmpow, frequency, osr=10):
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
    minfreq = max(0.0, best_freq - delf)
    maxfreq = best_freq + delf

    initial_guess = np.array([ampmax, best_freq, best_ph, 0.0])

    # (Aug 17, 2025) interval is extended to allow more flexibility
    minph = 0
    maxph = 1

    bounds = (
        [minamp, minfreq, minph, -np.inf],
        [maxamp, maxfreq, maxph, np.inf]
    )

    params, _ = curve_fit(
        sinusoid,
        time,
        flux,
        p0=initial_guess,
        bounds=bounds,
        max_nfev=100000
    )

    y_fit = sinusoid(time, *params)
    res_fit = y_fit - flux

    # Box indexes and freqs
    half_box = int(box/2)
    lind = max(0, indmpow - half_box)
    rind = min(len(frequency), indmpow + half_box + 1)
    fre = frequency[lind:rind]

    # Periodogram of the residuals
    nif = nifty_ls.lombscargle(
        time,
        res_fit,
        normalization='psd',
        center_data=True,
        fit_mean=False,
        fmin=fre[0],
        fmax=fre[-1],
        Nf=len(fre)
    )

    res_amps = 2*np.sqrt(nif.power/N)
    noise = np.median(res_amps)
    # Avoid invalid SNR values when the local noise estimate is zero
    # or non-finite.
    if not np.isfinite(noise) or noise <= 0:
        return noise, np.nan

    snr = params[0]/noise

    return noise, snr


def noise_spectrum(time, flux, osratio, kernel_size):
    """Estimation of noise spectrum using a median filter on the amplitude spectrum."""
    ls = LombScargle(
        time,
        flux,
        normalization='psd',
        center_data=True,
        fit_mean=False
    )

    _, power = ls.autopower(samples_per_peak=osratio)
    L = len(time)
    amps = 2*np.sqrt(power/L)

    if kernel_size % 2 == 0:  # fix for even kernel size
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
    ls = LombScargle(
        time,
        flux,
        normalization='psd',
        center_data=True,
        fit_mean=False
    )

    frequency, power = ls.autopower(
        maximum_frequency=max_freq,
        samples_per_peak=osratio,
        method="fastnifty"
    )

    indmpow = np.argmax(power)
    best_frequency = frequency[indmpow]
    amps = 2*np.sqrt(power/len(time))
    ampmax = amps[indmpow]

    return ls, frequency, amps, best_frequency, ampmax, power, indmpow


def fit(t, params):
    '''Multi-sine fit function with all the parameters of frequencies, amplitudes, phases'''
    pars = params.valuesdict()

    amps_dict = {
        k: pars[k]
        for k in pars
        if k.endswith('a')
    }

    freqs_dict = {
        k: pars[k]
        for k in pars
        if k.endswith('b')
    }

    phs_dict = {
        k: pars[k]
        for k in pars
        if k.endswith('c')
    }

    freqs = list(freqs_dict.values())
    amps = list(amps_dict.values())
    phs = list(phs_dict.values())

    # Floating mean: the constant term is added once per sinusoidal component
    b = pars.get('p_m', 0.0)

    y = np.zeros_like(t, dtype=float)

    for (a, f, p) in zip(amps, freqs, phs):
        y += sinusoid(t, a, f, p, b)

    return y, amps, freqs, phs


def residual(params, t, flux):
    '''Residual between the model and data'''
    res = fit(t, params)[0] - flux

    return res


def lightcurve(file, isfits, isascii, timecol, fluxcol, header_lines):
    '''Reading all data and removing non-finite samples.'''
    if isfits:
        with fits.open(file) as hdul:
            data = hdul[1].data
            data = pd.DataFrame(np.array(data))

        time = np.array(
            data.iloc[:, timecol-1],
            dtype=np.float64
        )

        fluxes = np.array(
            data.iloc[:, fluxcol-1],
            dtype=np.float64
        )

    elif isascii:
        data = np.loadtxt(
            file,
            skiprows=header_lines,
            ndmin=2
        )

        time = np.asarray(
            data[:, timecol-1],
            dtype=np.float64
        )

        fluxes = np.asarray(
            data[:, fluxcol-1],
            dtype=np.float64
        )

    else:
        raise ValueError('Input file type was not recognised.')

    finite = np.isfinite(time) & np.isfinite(fluxes)
    time = time[finite]
    fluxes = fluxes[finite]

    if len(time) < 3:
        raise ValueError('At least three finite time-series points are required.')

    # Ensure chronological order.
    order = np.argsort(time)
    time = time[order]
    fluxes = fluxes[order]

    time = time - time[0]

    T = time[-1] - time[0]
    N = len(time)

    if T <= 0:
        raise ValueError('Time span must be positive.')

    r = 1/T  # Rayleigh frequency resolution

    return time, fluxes, T, N, r


def snr_or_fap(par,
               min_snr=np.nan,
               S_N=np.nan,
               max_fap=np.nan,
               all_faps=np.nan):
    '''Choosing between the SNR or the FAP stop criterion.
       SNR by default'''
    if 'SNR' in par:
        return min_snr, S_N

    elif 'FAP' in par:
        return max_fap, all_faps

    else:
        raise ValueError("stop must be either 'SNR' or 'FAP'.")


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
    n2_list = np.arange(-2, 2).tolist()
    children1_2 = []

    n1_n2 = []

    for i in n1_list:
        for j in n2_list:
            candidate = i*f1 + j*f2
            if 0 < candidate < f_lim and candidate != f1 and candidate != f2:
                children1_2.append(round(i*f1 + j*f2, 2))
                n1_n2.append((i, j, 0))

    comb_freqs = []
    comb_amps = []
    n1_n2_values = []

    error = R

    for (ch, v) in zip(children1_2, n1_n2):
        for (f, a) in zip(freqs, amps):
            if abs(ch-f) < error:
                comb_freqs.append(f)
                comb_amps.append(a)
                n1_n2_values.append(v)

    return comb_freqs, comb_amps, n1_n2_values
"""


def select_resolved_peak(frequency, power, amps,
                         accepted_freqs, blocked_ranges, R):
    '''Select the highest-power peak resolved from accepted frequencies
    and outside previously rejected resolution intervals.'''
    
    # Consider only finite periodogram values.
    valid = (
        np.isfinite(frequency)
        & np.isfinite(power)
        & np.isfinite(amps)
    )
    valid_indices = np.flatnonzero(valid)

    if len(valid_indices) == 0:
        return None, None, None

    order = valid_indices[
        np.argsort(power[valid_indices])[::-1]
    ]

    for i in order:
        freq = frequency[i]

        # Must be resolved from all accepted frequencies.
        if any(abs(freq - f0) < R for f0 in accepted_freqs):
            continue

        # Must not lie inside a previously rejected resolution interval.
        if any(fmin <= freq <= fmax for fmin, fmax in blocked_ranges):
            continue

        return i, freq, amps[i]

    return None, None, None


def _check_resolved(fitted_freqs, fixed_freqs, R):
    '''Check Rayleigh resolution against fixed and current frequencies.'''

    fitted_freqs = np.asarray(fitted_freqs, dtype=float)

    if not np.all(np.isfinite(fitted_freqs)):
        return False

    # Check against frequencies already accepted in previous blocks.
    if any(
        abs(f - f0) < R
        for f in fitted_freqs
        for f0 in fixed_freqs
    ):
        return False

    # Check within the current simultaneous fit.
    if len(fitted_freqs) > 1:
        sorted_freqs = np.sort(fitted_freqs)

        if np.any(np.diff(sorted_freqs) < R):
            return False

    return True


def fit_resolved_candidate(params, time, flux, R, fixed_freqs, new_n):
    '''
    Perform a robust two-stage nonlinear least-squares fit.

    Stage 1 fits only the newly added component while the already fitted
    components remain fixed. Stage 2 releases all parameters and performs
    the full simultaneous fit.

    Several deterministic starting points are tried because a high-order
    multisine fit can otherwise fail simply because the optimizer starts in
    an unfavourable local basin.

    The candidate is accepted only when the final fitted frequencies are
    mutually separated by at least the Rayleigh resolution R and are also
    separated from frequencies accepted in previous blocks.
    '''

    new_amp_name = 'p_' + str(new_n) + 'a'
    new_freq_name = 'p_' + str(new_n) + 'b'
    new_ph_name = 'p_' + str(new_n) + 'c'

    base_freq = params[new_freq_name].value
    base_phase = params[new_ph_name].value

    freq_min = params[new_freq_name].min
    freq_max = params[new_freq_name].max
    ph_min = params[new_ph_name].min
    ph_max = params[new_ph_name].max

    # Deterministic multi-start seeds. The LS maximum remains the primary
    # starting point; additional seeds only help the optimizer escape a bad
    # local basin without introducing random behaviour.
    seeds = [
        (base_freq, base_phase),
        (np.clip(base_freq - R/4, freq_min, freq_max), base_phase),
        (np.clip(base_freq + R/4, freq_min, freq_max), base_phase),
        (base_freq, np.clip(base_phase + 0.5, ph_min, ph_max)),
        (base_freq, np.clip(base_phase - 0.5, ph_min, ph_max)),
    ]

    best_resolved = None
    best_resolved_params = None
    best_resolved_cost = np.inf

    best_unresolved = None
    best_unresolved_params = None
    best_unresolved_cost = np.inf

    for seed_freq, seed_phase in seeds:

        trial_params = params.copy()
        trial_params[new_freq_name].value = seed_freq
        trial_params[new_ph_name].value = seed_phase

        # ----------------------------------------------------------
        # Stage 1: fit only the new component.
        # ----------------------------------------------------------
        for name in trial_params:
            if name in {
                new_amp_name,
                new_freq_name,
                new_ph_name
            }:
                trial_params[name].vary = True

            elif new_n == 1 and name == 'p_m':
                trial_params[name].vary = True

            else:
                trial_params[name].vary = False

        res1 = minimize(
            residual,
            trial_params,
            args=(time, flux),
            method='least_squares',
            x_scale='jac',
            max_nfev=100000
        )

        if not res1.success:
            continue

        # ----------------------------------------------------------
        # Stage 2: release the complete simultaneous model.
        # ----------------------------------------------------------
        trial_params = res1.params.copy()

        for name in trial_params:
            trial_params[name].vary = True

        res2 = minimize(
            residual,
            trial_params,
            args=(time, flux),
            method='least_squares',
            x_scale='jac',
            max_nfev=100000
        )

        if not res2.success:
            continue

        # Reject solutions with non-finite residuals.
        if not np.all(np.isfinite(res2.residual)):
            continue

        fitted_params = res2.params
        fitted_freqs = np.asarray(
            fit(time, fitted_params)[2],
            dtype=float
        )

        if not np.all(np.isfinite(fitted_freqs)):
            continue

        cost = float(np.sum(np.asarray(res2.residual, dtype=float)**2))
        if not np.isfinite(cost):
            continue

        resolved = _check_resolved(
            fitted_freqs,
            fixed_freqs,
            R
        )

        if resolved:
            if cost < best_resolved_cost:
                best_resolved = res2
                best_resolved_params = fitted_params
                best_resolved_cost = cost

        else:
            if cost < best_unresolved_cost:
                best_unresolved = res2
                best_unresolved_params = fitted_params
                best_unresolved_cost = cost

    # Prefer the best converged solution that satisfies the resolution
    # criterion. If no resolved solution exists but a converged solution does,
    # return it so that the calling routine can reject the candidate as an
    # unresolved structure. If nothing converged, signal a numerical failure.
    if best_resolved is not None:
        return True, best_resolved, best_resolved_params

    if best_unresolved is not None:
        return False, best_unresolved, best_unresolved_params

    return False, None, None


def parameter_errors(params, residuals, N):
    '''
    Return formal 1-sigma uncertainties from the simultaneous fit.

    lmfit's covariance-based parameter errors are preferred because they
    include correlations among the simultaneously fitted parameters. If a
    standard error is unavailable, NaN is returned rather than silently
    substituting an unrelated quantity.
    '''

    amp_names = [
        k for k in params
        if k.endswith('a')
    ]

    freq_names = [
        k for k in params
        if k.endswith('b')
    ]

    ph_names = [
        k for k in params
        if k.endswith('c')
    ]

    sigma_amps = [
        params[k].stderr
        if params[k].stderr is not None
        else np.nan
        for k in amp_names
    ]

    sigma_freqs = [
        params[k].stderr
        if params[k].stderr is not None
        else np.nan
        for k in freq_names
    ]

    sigma_phs = [
        params[k].stderr
        if params[k].stderr is not None
        else np.nan
        for k in ph_names
    ]

    return sigma_amps, sigma_freqs, sigma_phs


def arguments(agrv):
    '''Function to parse the command line arguments'''

    parser = argparse.ArgumentParser(
        description='Open the directory with the files to be processed'
    )

    parser.add_argument(
        '--p',
        type=str,
        help='Select a parameters file'
    )

    parser.add_argument(
        '--d',
        type=str,
        help='Select a directory containing a list of files'
    )

    parser.add_argument(
        '--file',
        type=str,
        help='Select a single file to be opened'
    )

    parser.add_argument(
        '-m',
        '--multiprocessing',
        action='store_true',
        required=False,
        help='Use multiprocessing. Requires --d.'
    )

    parser.add_argument(
        '-np',
        '--num_processes',
        type=int,
        required=False,
        help='Number of processes to use.'
    )

    args = parser.parse_args(agrv)

    return args


# Here begins the main part of the code
def multimodes(args, multipro_input=None,dash=100*'-'):

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
    header_lines = 1  # Number of header lines in ASCII input files
    #save_plot_per = 0  # save plots of periodogram every xx iterations
    save_data_res = 0  # save data of residual every xx iterations
    save_freq_list = 0  # save result files (freq.list) every xx iterations (must be a multiple of sim_fit_n)
    save_plot_resps = 0  # write residual spectrum after last iteration
    max_iter = 1000  # Multimodes
    clean_close = 0  # kept for backwards compatibility; resolution is enforced during extraction
    noise_method = 'hf'  # method to estimate the noise level

    if os.path.isfile(paramname):

        with open(paramname, 'r') as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                if line.startswith("sim_fit_n"):
                    sim_fit_n = int(line.split()[1])

                if line.startswith("osratio"):
                    osratio = float(line.split()[1])

                if line.startswith("max_freq"):
                    # Accept either a numerical maximum frequency or the Nyquist keyword.
                    max_freq_value = line.split()[1]

                    if max_freq_value.lower() == "nyquist":
                        max_freq = "nyquist"
                    else:
                        max_freq = float(max_freq_value)

                if line.startswith("max_fap"):
                    max_fap = float(line.split()[1])

                if line.startswith("min_snr"):
                    min_snr = float(line.split()[1])

                if line.startswith("stop"):
                    stop = str(line.split()[1])

                if line.startswith("timecol"):
                    timecol = int(line.split()[1])

                if line.startswith("fluxcol"):
                    fluxcol = int(line.split()[1])

                if line.startswith("save_data_res"):
                    save_data_res = int(line.split()[1])

                if line.startswith("save_plot_resps"):
                    save_plot_resps = int(line.split()[1])

                if line.startswith("header_lines"):
                    header_lines = int(line.split()[1])

                if line.startswith("max_iter"):
                    max_iter = int(line.split()[1])

                    if max_iter == 0:
                        max_iter = 1e6

                if line.startswith("save_freq_list"):
                    save_freq_list = int(line.split()[1])

                if line.startswith("clean_close"):
                    clean_close = int(line.split()[1])

                if line.startswith("noise_method"):
                    noise_method = line.split()[1]

    else:
        print('No param file. Default values will be used:')

    if stop not in ('SNR', 'FAP'):
        raise ValueError("stop must be either 'SNR' or 'FAP'.")

    if noise_method not in ('hf', 'g', 'box'):
        raise ValueError("noise_method must be 'hf', 'g' or 'box'.")

    if osratio <= 0:
        raise ValueError('osratio must be positive.')
    
    # Maximum frequency validation.
    # Nyquist is calculated separately for each light curve.
    if isinstance(max_freq, str):
    
        if max_freq != "nyquist":
            raise ValueError(
                "max_freq must be a positive number or 'Nyquist'."
            )
    
    else:
    
        if not np.isfinite(max_freq) or max_freq <= 0:
            raise ValueError('max_freq must be positive.')

    if sim_fit_n < 1:
        raise ValueError('sim_fit_n must be at least 1.')

    if max_iter < 0:
        raise ValueError('max_iter cannot be negative.')

    # clean_close is intentionally ignored. Resolution is enforced during
    # extraction and fitting, so a late post-processing filter is redundant.
    if clean_close != 0:
        print('Warning: clean_close is ignored; Rayleigh resolution is enforced during extraction.')

    print(
        'Number of frequencies of the simultaneous fit: '
        + str(sim_fit_n)
    )

    print(
        'Samples per peak: '
        + str(osratio)
    )

    print(
        'Maximum frequency: '
        + str(max_freq)
    )

    if stop == 'SNR':
        print(
            'Stop Criterion: SNR > '
            + str(min_snr)
        )

    elif stop == 'FAP':
        print(
            'Stop Criterion: FAP < '
            + str(max_fap)
        )

    # Creating the list with all the files
    isfits = False
    isascii = False
    
    if args.multiprocessing:
    
        if not args.d:
            raise ValueError(
                'Multiprocessing requires --d.'
            )
    
        if multipro_input is None:
            raise ValueError(
                'No input file was provided to the worker process.'
            )
    
        pth = os.path.abspath(os.path.expanduser(args.d))
    
        # The worker receives one concrete light-curve file.
        filepath = [multipro_input]
    
        rel_path = os.path.relpath(
            multipro_input,
            pth
        )
    
        fname = [
            os.path.splitext(rel_path)[0]
        ]
    
        if multipro_input.endswith('.fits'):
            isfits = True
    
        elif (
            multipro_input.endswith('.dat')
            or multipro_input.endswith('.txt')
        ):
            isascii = True
    
        else:
            raise ValueError(
                'Input file must have .fits, .dat or .txt extension.'
            )
    
    elif args.d:
    
        pth = os.path.abspath(os.path.expanduser(args.d))

        fits_files = sorted(
            glob.glob(os.path.join(pth, '*.fits'))
        )

        ascii_files = sorted(
            glob.glob(os.path.join(pth, '*.dat'))
        )
    
        ascii_files = [
            f for f in glob.glob(pth + '*.dat')
        ]
    
        if len(ascii_files) == 0:
            ascii_files = sorted(
                glob.glob(os.path.join(pth, '*.txt'))
            )
    
        if len(ascii_files) == 0:
        
            if len(fits_files) == 0:
                raise FileNotFoundError(
                    "No input files found in "+ pth
                )
    
            filepath = fits_files
            isfits = True
    
        else:

            filepath = ascii_files
            isascii = True

        # Extracting file names after sorting the input paths.
        fname = [
            os.path.splitext(f)[0]
            for f in filepath
        ]
    else:

        if not args.file:
            raise ValueError(
                'Either --d or --file must be provided.'
            )

        # Expand home directory and resolve relative paths.
        input_file = os.path.abspath(
            os.path.expanduser(args.file)
        )

        # Check that the input file exists.
        if not os.path.isfile(input_file):
            raise FileNotFoundError(
                f'Input file not found: {input_file}'
            )

        # Identify the input file format.
        extension = os.path.splitext(input_file)[1].lower()

        if extension == '.fits':

            filepath = [input_file]
            isfits = True

        elif extension in ('.dat', '.txt'):

            filepath = [input_file]
            isascii = True

        else:

            raise ValueError(
                'Input file must have .fits, .dat or .txt extension.'
            )

        # Use only the file name, without its extension,
        # to construct the output directory.
        fname = [
            os.path.splitext(os.path.basename(input_file))[0]
        ]

    # Starting with the iterative analysis
    columns = [
        'Number',
        'f',
        'A',
        stop
    ]

    # Setting stop criterion
    parade = snr_or_fap(
        stop,
        min_snr=min_snr,
        S_N=S_N
    )[0]

    snr_or_faps = snr_or_fap(
        stop,
        max_fap=max_fap,
        all_faps=all_faps
    )[1]

    # Initialization of the parameter structure
    params = Parameters()

    for (f, nm) in zip(filepath, fname):

        start = timer()

        n = 1
        num = 1

        data = lightcurve(
            f,
            isascii=isascii,
            isfits=isfits,
            timecol=timecol,
            fluxcol=fluxcol,
            header_lines=header_lines
        )

        time = data[0]
        lc = data[1]
        # Computing the maximum frequency independently for each light curve.
        # For Nyquist mode, the median positive sampling interval is used.
        if max_freq == "nyquist":
        
            cadence = np.diff(time)
            cadence = cadence[np.isfinite(cadence) & (cadence > 0)]
        
            if len(cadence) == 0:
                raise ValueError(
                    'Cannot calculate Nyquist frequency: '
                    'no positive sampling intervals were found.'
                )
        
            max_freq_lc = 1.0 / (2.0 * np.median(cadence))
        
        else:
        
            max_freq_lc = float(max_freq)

        T = data[2]
        N = data[3]
        R = data[4]

        # ==============================================================
        # DEFINE OUTPUT PATH
        # ==============================================================

        if args.d:
        
            # Get the input directory name independently of the path format.
            input_dir_name = os.path.basename(
                os.path.normpath(
                    os.path.abspath(os.path.expanduser(args.d))
                )
            )

            # Create the main output directory.
            if os.path.isfile(paramname):
            
                parameter_name = os.path.splitext(
                    os.path.basename(paramname)
                )[0]

                ofolder = os.path.join(
                    './results',
                    input_dir_name + '+' + parameter_name
                )

            else:
            
                ofolder = os.path.join(
                    './results',
                    input_dir_name
                )

            # Each light curve gets its own output directory.
            if args.multiprocessing:
            
                # nm is relative to the input directory in multiprocessing.
                newpath = os.path.join(
                    ofolder,
                    nm,
                    ''
                )

            else:
            
                # nm contains the full input path in sequential mode.
                # Only the light-curve filename is used for the output folder.
                newpath = os.path.join(
                    ofolder,
                    os.path.basename(nm),
                    ''
                )

        else:
        
            # ==============================================================
            # SINGLE FILE OUTPUT PATH
            # ==============================================================

            # Get the light-curve filename without its extension.
            lightcurve_name = os.path.basename(nm)

            if os.path.isfile(paramname):
            
                # Get parameter file name independently of its path.
                parameter_name = os.path.splitext(
                    os.path.basename(paramname)
                )[0]

                # Main output directory.
                ofolder = os.path.join(
                    './results',
                    '+' + parameter_name
                )

                newpath = os.path.join(
                    ofolder,
                    lightcurve_name,
                    ''
                )

            else:
            
                newpath = os.path.join(
                    './results',
                    lightcurve_name,
                    ''
                )
        """    
        # Define the path for the output
        if os.path.isfile(paramname):
        
            if args.d:
            
                ofolder = (
                    './results/'
                    + pth[2:-1]
                    + '+'
                    + paramname[:-4]
                    + '/'
                )
        
                if args.multiprocessing:
                
                    newpath = (
                        ofolder
                        + nm
                        + '/'
                    )
        
                else:
                
                    filesp = nm[2:].split('/')
        
                    file = filesp[1]
        
                    newpath = (
                        ofolder
                        + file
                        + '/'
                    )
        
            else:
            
                ofolder = (
                    './results/+'
                    + paramname[:-4]
                    + '/'
                )
        
                newpath = (
                    ofolder
                    + nm
                    + '/'
                )
        
        else:
        
            if args.multiprocessing:
            
                newpath = (
                    './results/'
                    + nm
                    + '/'
                )
        
            else:
            
                newpath = (
                    './results/'
                    + nm
                    + '/'
                )
        """

        if not os.path.exists(newpath):
            os.makedirs(newpath)

        # Calculating the initial periodogram
        lc0 = lc - np.mean(lc)
        lc = lc0

        # Noise level estimation
        if noise_method == "hf":
            no = hf_noise_estimate(
                time,
                lc,
                osratio
            )

        elif noise_method == "g":
            no = global_noise_estimate(
                time,
                lc,
                osratio
            )

        else:
            no = np.nan

        # Initialization of the lists to save the extracted frequencies,
        # amplitudes and phases
        all_best_freqs = []
        all_max_amps = []
        all_phs = []
        snr_or_faps = []
        all_sigma_amps = []
        all_sigma_freqs = []
        all_sigma_phs = []
        all_rms = []

        # Frequencies already accepted
        accepted_freqs = []

        # Intervals corresponding to rejected candidates
        blocked_ranges = []

        params = Parameters()
        last_ls = None
        sigma_amps = []
        sigma_freqs = []
        sigma_phs = []

        # Print header table
        print(dash)

        print(
            '{:<15s}{:>18s}{:>25s}{:>32}'
            .format(
                columns[0],
                columns[1],
                columns[2],
                columns[3]
            )
        )

        print(dash)

        while n <= sim_fit_n:

            ls = periodogram(
                time,
                lc,
                osratio=osratio,
                max_freq=max_freq_lc
            )

            last_ls = ls

            frequency = ls[1]
            amps = ls[2]
            power = ls[5]

            imax, freq, amp = select_resolved_peak(
                frequency,
                power,
                amps,
                accepted_freqs,
                blocked_ranges,
                R
            )

            if imax is None:
            
                # No more resolved peaks are available.
                # Save the frequencies already extracted in the current
                # simultaneous-fit block before stopping the extraction.
                if n != 1:
                
                    block_freqs = fit(time, params)[2]
                    block_amps = fit(time, params)[1]
                    block_phs = fit(time, params)[3]

                    all_best_freqs += block_freqs
                    all_max_amps += block_amps
                    all_phs += block_phs

                    # Formal uncertainties corresponding to the current block.
                    all_sigma_amps += sigma_amps
                    all_sigma_freqs += sigma_freqs
                    all_sigma_phs += sigma_phs

                break

            rms = np.sqrt(
                np.sum(lc**2)/N
            )

            if noise_method == "box":

                no, snr = box_noise_estimate(
                    time,
                    lc,
                    freq,
                    amp,
                    imax,
                    ls[1],
                    osr=osratio
                )

            else:
                # A valid positive noise estimate is required to calculate SNR.
                if np.isfinite(no) and no > 0:
                    snr = amp/no
                else:
                    snr = np.nan

            var = (N/4)*no**2

            # FAP is calculated from the selected peak, not the absolute
            # maximum of the periodogram.
            if np.isfinite(var) and var > 0:
                fap = 1 - (
                    1 - np.exp(-power[imax]/var)
                )**(N/2)
            else:
                fap = np.nan

            # ==========================================================
            # SNR STOP CRITERION
            # ==========================================================

            if stop == 'SNR':

                if snr > parade and num <= max_iter:

                    # Save the previous model in case the new candidate
                    # has to be rejected.
                    params_before = params.copy()
                    lc_before = lc.copy()

                    # Non-linear least-squares fitting
                    minamp = 0.0
                    maxamp = np.inf

                    delf = R/2

                    minfreq = max(
                        0.0,
                        freq - delf
                    )

                    maxfreq = min(
                        max_freq_lc,
                        freq + delf
                    )

                    if maxfreq <= minfreq:
                        blocked_ranges.append(
                            (freq - 0.5*np.median(np.diff(frequency)),
                             freq + 0.5*np.median(np.diff(frequency)))
                        )
                        continue

                    ph = initial_phase(
                        time,
                        lc,
                        freq,
                        ofac=osratio
                    )

                    # Phase is represented in cycles.
                    minph = -0.5
                    maxph = 1.5

                    # Add the new candidate to the simultaneous model.
                    params.add(
                        'p_'+str(n)+'a',
                        value=amp,
                        min=minamp,
                        max=maxamp
                    )

                    params.add(
                        'p_'+str(n)+'b',
                        value=freq,
                        min=minfreq,
                        max=maxfreq
                    )

                    params.add(
                        'p_'+str(n)+'c',
                        value=np.clip(ph, minph, maxph),
                        min=minph,
                        max=maxph
                    )

                    # Floating mean, added only once.
                    if n == 1:
                        params.add(
                            'p_m',
                            value=0.0
                        )

                    # Simultaneous nonlinear least-squares fit.
                    resolved, res, fitted_params = fit_resolved_candidate(
                        params,
                        time,
                        lc0,
                        R,
                        all_best_freqs,
                        n
                    )

                    # Reject a candidate if no converged resolved solution
                    # exists, restoring the previous model.
                    if not resolved:

                        params = params_before
                        lc = lc_before

                        if res is None:

                            # Numerical failure: block only the current LS
                            # grid cell, not the whole Rayleigh interval.
                            df_grid = np.median(np.diff(frequency))

                            blocked_ranges.append(
                                (
                                    freq - 0.5*df_grid,
                                    freq + 0.5*df_grid
                                )
                            )

                            print(
                                'Candidate rejected: simultaneous '
                                'fit did not converge.'
                            )

                        else:

                            # A converged fit exists, but it is unresolved.
                            # The accepted fitted frequencies already define
                            # the Rayleigh exclusion region. Therefore, do not
                            # block an additional physical interval here.
                            # Block only the current LS grid cell so that the
                            # same candidate is not selected again.
                            df_grid = np.median(np.diff(frequency))

                            blocked_ranges.append(
                                (
                                    freq - 0.5*df_grid,
                                    freq + 0.5*df_grid
                                )
                            )

                            print(
                                'Candidate rejected: fitted frequencies '
                                'are closer than the Rayleigh resolution.'
                            )

                        continue

                    # Accepted simultaneous solution.
                    params = fitted_params
                    lc = res.residual

                    # The residual has changed after accepting the new frequency.
                    # Previously rejected candidates may therefore be reconsidered.
                    blocked_ranges = []

                    fitted_freqs = fit(
                        time,
                        params
                    )[2]

                    accepted_freqs = (
                        all_best_freqs
                        + fitted_freqs
                    )

                    # Formal uncertainties from the final simultaneous fit.
                    sigma_amps, sigma_freqs, sigma_phs = parameter_errors(
                        params,
                        lc,
                        N
                    )

                    # Save detection information only for accepted candidates.
                    snr_or_faps.append(snr)
                    all_rms.append(rms)

                    # Print accepted frequency. The displayed f and A here
                    # are the LS candidate values; best_modes.dat stores the
                    # final simultaneous-fit values.
                    print(
                        '{:<15d}{:>18f}{:>25f}{:>32}'
                        .format(
                            num,
                            freq,
                            amp,
                            snr
                        )
                    )

                    # Save residuals
                    if save_data_res != 0:

                        if np.mod(
                            num,
                            save_data_res
                        ) == 0:

                            lc_df = pd.DataFrame({
                                'Time': time,
                                'Flux': list(lc)
                            })

                            lc_df.to_csv(
                                newpath
                                + 'res_'
                                + str(num)
                                + '.dat',
                                sep=' ',
                                index=False,
                                header=None
                            )

                    n += 1
                    num += 1

                    # ==================================================
                    # END OF SIMULTANEOUS FIT BLOCK
                    # ==================================================

                    if n > sim_fit_n:

                        block_freqs = fit(time, params)[2]
                        block_amps = fit(time, params)[1]
                        block_phs = fit(time, params)[3]

                        all_best_freqs += block_freqs
                        all_max_amps += block_amps
                        all_phs += block_phs
                        all_sigma_amps += sigma_amps
                        all_sigma_freqs += sigma_freqs
                        all_sigma_phs += sigma_phs

                        lc0 = lc - np.mean(lc)

                        # Write the frequency list
                        if save_freq_list != 0:

                            if np.mod(
                                num,
                                save_freq_list
                            ) == 0:

                                prew_df = pd.DataFrame({
                                    'Freqs': all_best_freqs,
                                    'Amps': all_max_amps,
                                    'Phases': all_phs,
                                    'Amplitude 1-sigma error (mmag)': all_sigma_amps,
                                    'Frequency 1-sigma error (c/d)': all_sigma_freqs,
                                    'Phase 1-sigma error (cycles)': all_sigma_phs,
                                    'SNR/FAP': snr_or_faps,
                                    'rms': all_rms
                                })

                                prew_df.to_csv(
                                    newpath
                                    + 'freqs_'
                                    + str(num-1)
                                    + '.dat',
                                    sep=' ',
                                    index=False,
                                    header=None
                                )

                        n = 1
                        params = Parameters()
                        accepted_freqs = all_best_freqs.copy()
                        blocked_ranges = []

                else:

                    # Stop before reaching sim_fit_n.
                    if n != 1:

                        block_freqs = fit(time, params)[2]
                        block_amps = fit(time, params)[1]
                        block_phs = fit(time, params)[3]

                        all_best_freqs += block_freqs
                        all_max_amps += block_amps
                        all_phs += block_phs
                        all_sigma_amps += sigma_amps
                        all_sigma_freqs += sigma_freqs
                        all_sigma_phs += sigma_phs

                    break

            # ==========================================================
            # FAP STOP CRITERION
            # ==========================================================

            elif stop == 'FAP':

                if fap < parade and num <= max_iter:

                    # Save the previous model in case the new candidate
                    # has to be rejected.
                    params_before = params.copy()
                    lc_before = lc.copy()

                    # Non-linear least-squares fitting
                    minamp = 0.0
                    maxamp = np.inf

                    delf = R/2

                    minfreq = max(
                        0.0,
                        freq - delf
                    )

                    maxfreq = min(
                        max_freq_lc,
                        freq + delf
                    )

                    if maxfreq <= minfreq:
                        blocked_ranges.append(
                            (freq - 0.5*np.median(np.diff(frequency)),
                             freq + 0.5*np.median(np.diff(frequency)))
                        )
                        continue

                    ph = initial_phase(
                        time,
                        lc,
                        freq,
                        ofac=osratio
                    )

                    # Phase is represented in cycles.
                    minph = -0.5
                    maxph = 1.5

                    # Add the new candidate to the simultaneous model.
                    params.add(
                        'p_'+str(n)+'a',
                        value=amp,
                        min=minamp,
                        max=maxamp
                    )

                    params.add(
                        'p_'+str(n)+'b',
                        value=freq,
                        min=minfreq,
                        max=maxfreq
                    )

                    params.add(
                        'p_'+str(n)+'c',
                        value=np.clip(ph, minph, maxph),
                        min=minph,
                        max=maxph
                    )

                    # Floating mean, added only once.
                    if n == 1:
                        params.add(
                            'p_m',
                            value=0.0
                        )

                    # Simultaneous nonlinear least-squares fit.
                    resolved, res, fitted_params = fit_resolved_candidate(
                        params,
                        time,
                        lc0,
                        R,
                        all_best_freqs,
                        n
                    )

                    # Reject a candidate if no converged resolved solution
                    # exists, restoring the previous model.
                    if not resolved:

                        params = params_before
                        lc = lc_before

                        if res is None:

                            # Numerical failure: block only the current LS
                            # grid cell, not the whole Rayleigh interval.
                            df_grid = np.median(np.diff(frequency))

                            blocked_ranges.append(
                                (
                                    freq - 0.5*df_grid,
                                    freq + 0.5*df_grid
                                )
                            )

                            print(
                                'Candidate rejected: simultaneous '
                                'fit did not converge.'
                            )

                        else:

                            # A converged fit exists, but it is unresolved.
                            # The accepted fitted frequencies already define
                            # the Rayleigh exclusion region. Therefore, do not
                            # block an additional physical interval here.
                            # Block only the current LS grid cell so that the
                            # same candidate is not selected again.
                            df_grid = np.median(np.diff(frequency))

                            blocked_ranges.append(
                                (
                                    freq - 0.5*df_grid,
                                    freq + 0.5*df_grid
                                )
                            )

                            print(
                                'Candidate rejected: fitted frequencies '
                                'are closer than the Rayleigh resolution.'
                            )

                        continue

                    # Accepted simultaneous solution.
                    params = fitted_params
                    lc = res.residual

                    # The residual has changed after accepting the new frequency.
                    # Previously rejected candidates may therefore be reconsidered.
                    blocked_ranges = []

                    fitted_freqs = fit(
                        time,
                        params
                    )[2]

                    accepted_freqs = (
                        all_best_freqs
                        + fitted_freqs
                    )

                    # Formal uncertainties from the final simultaneous fit.
                    sigma_amps, sigma_freqs, sigma_phs = parameter_errors(
                        params,
                        lc,
                        N
                    )

                    # Save detection information only for accepted candidates.
                    snr_or_faps.append(fap)
                    all_rms.append(rms)

                    # Print accepted frequency. The displayed f and A here
                    # are the LS candidate values; best_modes.dat stores the
                    # final simultaneous-fit values.
                    print(
                        '{:<15d}{:>18f}{:>25f}{:>32}'
                        .format(
                            num,
                            freq,
                            amp,
                            fap
                        )
                    )

                    # Save residuals
                    if save_data_res != 0:

                        if np.mod(
                            num,
                            save_data_res
                        ) == 0:

                            lc_df = pd.DataFrame({
                                'Time': time,
                                'Flux': list(lc)
                            })

                            lc_df.to_csv(
                                newpath
                                + 'res_'
                                + str(num)
                                + '.dat',
                                sep=' ',
                                index=False,
                                header=None
                            )

                    n += 1
                    num += 1

                    # ==================================================
                    # END OF SIMULTANEOUS FIT BLOCK
                    # ==================================================

                    if n > sim_fit_n:

                        block_freqs = fit(time, params)[2]
                        block_amps = fit(time, params)[1]
                        block_phs = fit(time, params)[3]

                        all_best_freqs += block_freqs
                        all_max_amps += block_amps
                        all_phs += block_phs
                        all_sigma_amps += sigma_amps
                        all_sigma_freqs += sigma_freqs
                        all_sigma_phs += sigma_phs

                        lc0 = lc - np.mean(lc)

                        # Write the frequency list
                        if save_freq_list != 0:

                            if np.mod(
                                num,
                                save_freq_list
                            ) == 0:

                                prew_df = pd.DataFrame({
                                    'Freqs': all_best_freqs,
                                    'Amps': all_max_amps,
                                    'Phases': all_phs,
                                    'Amplitude 1-sigma error (mmag)': all_sigma_amps,
                                    'Frequency 1-sigma error (c/d)': all_sigma_freqs,
                                    'Phase 1-sigma error (cycles)': all_sigma_phs,
                                    'SNR/FAP': snr_or_faps,
                                    'rms': all_rms
                                })

                                prew_df.to_csv(
                                    newpath
                                    + 'freqs_'
                                    + str(num-1)
                                    + '.dat',
                                    sep=' ',
                                    index=False,
                                    header=None
                                )

                        n = 1
                        params = Parameters()
                        accepted_freqs = all_best_freqs.copy()
                        blocked_ranges = []

                else:

                    # Stop before reaching sim_fit_n.
                    if n != 1:

                        block_freqs = fit(time, params)[2]
                        block_amps = fit(time, params)[1]
                        block_phs = fit(time, params)[3]

                        all_best_freqs += block_freqs
                        all_max_amps += block_amps
                        all_phs += block_phs
                        all_sigma_amps += sigma_amps
                        all_sigma_freqs += sigma_freqs
                        all_sigma_phs += sigma_phs

                    break

        # ==============================================================
        # FINAL DATAFRAME
        # ==============================================================

        prew_df = pd.DataFrame({
            'Freqs': all_best_freqs,
            'Amps': all_max_amps,
            'Phases': all_phs,
            'Amplitude 1-sigma error (mmag)': all_sigma_amps,
            'Frequency 1-sigma error (c/d)': all_sigma_freqs,
            'Phase 1-sigma error (cycles)': all_sigma_phs,
            'SNR/FAP': snr_or_faps,
            'rms': all_rms
        })

        # Resolution is already enforced during extraction.
        # clean_close is therefore intentionally not applied here.

        # ==============================================================
        # WRITE FINAL RESIDUAL LIGHT CURVE
        # ==============================================================

        reslc = pd.DataFrame({
            'Time': time,
            'Residuals': lc
        })

        reslc.to_csv(
            newpath + 'res_lc.dat',
            sep=' ',
            index=False,
            header=None
        )

        # ==============================================================
        # WRITE FINAL FREQUENCY LIST
        # ==============================================================

        prew_df.to_csv(
            newpath + 'best_modes.dat',
            sep=' ',
            index=False,
            header=None
        )

        # ==============================================================
        # SAVE RESIDUAL SPECTRUM
        # ==============================================================

        if save_plot_resps != 0 and last_ls is not None:

            resps = pd.DataFrame({
                'Frequencies': last_ls[1],
                'Amplitudes': last_ls[2]
            })

            resps.to_csv(
                newpath + 'res_ps.dat',
                sep=' ',
                index=False,
                header=None
            )

        end = timer()

        print(
            'Executing Time: '
            + str(end-start)
        )

    return



def multimodes_wrapper(packed_args):
    '''
    Wrapper for multiprocessing.

    Each worker receives the global argparse Namespace and one
    individual light-curve path.
    '''
    args, multipro_input = packed_args
    
    try:
        return multimodes(
            args,
            multipro_input=multipro_input
        )

    except Exception as exc:
        import traceback

        print(
            f"\n[ERROR] Fallo procesando la tarea: {args}",
            flush=True
        )
        print(
            f"[ERROR] {type(exc).__name__}: {exc}",
            flush=True
        )
        traceback.print_exc()

        return None


# =========================================

if __name__ == "__main__":

    import sys
    import multiprocessing as mp

    dash = 100*'-'

    print(dash)
    print('Running MultiModes'.center(110))
    print(dash)

    # Reading the arguments.
    args = arguments(
        agrv=sys.argv[1:]
    )

    if args.multiprocessing:

        if not args.d:
            raise ValueError(
                'Multiprocessing requires --d.'
            )

        total_cores = os.cpu_count() or 1

        # Manual number of processes.
        if args.num_processes is not None:

            safe_cores = max(
                1,
                min(
                    args.num_processes,
                    total_cores
                )
            )

            print(
                f'[*] MOTOR HPC: Using '
                f'{safe_cores} processes '
                f'(requested: {args.num_processes}, '
                f'available: {total_cores}).'
            )

        # Automatic 80% rule.
        else:

            safe_cores = max(
                1,
                int(total_cores * 0.8)
            )

            print(
                f'[*] MOTOR HPC: Using '
                f'{safe_cores} of '
                f'{total_cores} available cores '
                f'(80% rule).'
            )

        # ----------------------------------------------------------
        # Recursive search for input light curves.
        # ----------------------------------------------------------

        # Absolute input directory, supporting relative, absolute and home paths.
        input_dir = os.path.abspath(
            os.path.expanduser(args.d)
        )

        # Recursive search for input light curves.
        lc_files = glob.glob(
            os.path.join(input_dir, '**', '*.fits'),
            recursive=True
        )

        if not lc_files:
        
            lc_files = glob.glob(
                os.path.join(input_dir, '**', '*.dat'),
                recursive=True
            )

        if not lc_files:
        
            lc_files = glob.glob(
                os.path.join(input_dir, '**', '*.txt'),
                recursive=True
            )

        if not lc_files:

            raise FileNotFoundError(
                'No input light-curve files were found in '
                + args.d
            )

        print(
            f'[*] Found {len(lc_files)} input files.'
        )

        # One task = one complete light-curve analysis.
        input_data = [
            (args, lc_file)
            for lc_file in lc_files
        ]

        # ----------------------------------------------------------
        # Small chunks are preferable because individual stars can
        # have very different execution times.
        # ----------------------------------------------------------

        chunksize = 1

        print(
            f'[*] Parallel execution enabled '
            f'(processes={safe_cores}, chunksize={chunksize}).'
        )

        with mp.Pool(
            processes=safe_cores
        ) as pool:

            for _ in pool.imap_unordered(
                multimodes_wrapper,
                input_data,
                chunksize=chunksize
            ):
                pass

    else:

        multimodes(args)

    print(dash)
    print('Finished'.center(110))
    print(dash)
