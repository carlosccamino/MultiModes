## What is it?
MultiModes2 is a Python code to extract the most significant frequencies of a set of lightcurve (or other time series) files contained in a folder.
The original Multimodes code is authored by David Pamos Ortega (University of Granada).  

Modifications by Javier Pascual Granado (IAA-CSIC) are aimed to make the code more efficient, accesible and extensible.

Other contributors: Antonio García Hernández (UGR), Sebastiano de Franciscis (IAA-CSIC) and Cristian Rodrigo.

## Description
MultiModes2 takes as input a directory (or file) with light curves, which are assumed to be corrected from trends, outliers and 'nan' values. The files can be in fits or ascii format. It also requires the initial parameters to be written in a text file (named ini.txt by default).

With every light curve, the code calculates the frequencies spectrum, or periodogram, with the Fast Lomb Scargle algorithm (Press & Ribicky 1989). It extracts the higher amplitude peak and evaluates if it is real signal or due to noise, either by the False Alarm Probability or by the Signal to Noise criterion, it is a decision of the user at the time of choosing the initial parameters. By default it is chosen to use as stop criterion that S/N is greater than 4, (Breger 1993) but this can be modified for a more suited choice. Then, Multimodes fits frequency, amplitude and phase through non-linear optimization, using a multisine function. This function is redefined with the new calculated parameters at each iteration. It does a simultaneous fit of a number of peaks (20 by default). Then, they are subtracted from the original signal and goes back to the beginning of the loop with the residual, repeating the same process, until the stop criterion is reached. 
 
Multimodes make use of [astropy](https://www.astropy.org) for the calculation of the periodograms and [lmfit](https://lmfit.github.io/lmfit-py/) for the non-linear and simultaneous fitting of the extracted signals, using the non-linear least squares method for python.

## Citations
If MultiModes2 is used, please, cite the authors this way: Pamos Ortega, D. et al. 2022 (https://doi.org/10.1093/mnras/stac864)

## Requirements
Python >=3.8 with the following modules installed:
- numpy >=1.19.2
- matplotlib >=3.3.2
- pandas >=1.1.2
- astropy >=4.0.2
- lmfit >=1.0.2
- scipy >=1.5.2

## Input
- Directory with light curves in fits or ASCII format (.dat or .txt), corrected from 'outliers' and 'nan' values. Note that the first row (header) in ASCII files is dropped by default.
- ini.txt with optional parameters

## Optional parameters:
- sim_fit_n (20 by default): Number of simultaneous peaks to be fitted before extracting to the original light curve for obtaining the residual
- max_freq (100 by default): Maximum value of the analysed frequencies
- osratio (5 by default): oversampling factor. Compare with low, medium and high (10, 15, 20x) in Period04. 
- stop (SNR by default): Stop criterion, FAP or SNR.
- min_snr (4 by default, Breger 1993): Minimum signal to noise ratio to detect a frequency
- max_fap (0.01 by default, Balona et al. 2014): Maximum value of the False Alarm Probability to detect a frequency.
- timecol (default is 1): column for time 
- fluxcol (default is 2): column for fluxes
- save_data_res (default is 0): save data of residual every xx iterations
- save_freq_list (default is 0): parameter allows to save intermediate results files each xx iterations (must be a multiple of sim_fit_n).
- save_plot_resps (default is 0): write residual spectrum after last iteration (binary flag)
- max_iter (default is 1000): maximum number of iterations
- header_lines (default is 1): skip xx header lines
- clean_close (0 by default): remove the less significant frequencies that are closer than Rayleigh (i.e. might be spurious). This might be useful only when osratio>1.
- noise_method ('hf' by default): allows to choose the method to estimate the noise level. Three options are possible: 'hf' is based on the mean amplitude in the range of highest frequencies (where no signal is expected). 'g' is based on the median amplitude along the whole spectrum, and 'box' is based on the median of the background inside a box centered around the peak of maximum amplitude.

## Output
- Directory 'results', containing subdirectories corresponding to every analysed light curve. Each subdirectory contains:
  - file best_modes.dat with 8 columns containing the values of the most significant frequencies, amplitudes, phases, the corresponding errors, SNR/FAP and the rms
  - res_lc.dat (optional), with the final residual after extracting all the most significant frequencies
  - Other intermediate residual files if save_data_res and/or save_freq_list parameter are activated
  - res_ps.dat (optional), with the periodogram of the final residual after extracting all the most significant frequencies.
  
Screen output shows the parameters for the peak at maximum amplitude of the Lomb-Scargle periodogram for each iteration.  
**NOTE:** These parameters might change at every iteration but the screen ouput just show the last frequency detected in the periodogram. The correct frequency parameter can only be seen in the result files, the screen output is just for reference.  

# How to run it
- Copy MultiModes2.py and ini.txt to your working directory.
- Put also the directory or file with the light curves to be analysed inside your working directory.
- Enter your working directory and type the command: 
`python MultiModes2.py --d <lightcurves_directory>`
or alternatively
`python MultiModes2.py --file <lightcurve_file>`

If you want to use other parameter file than ini.txt you can call the program using the flag --p <ini_file>. For example, 
`python MultiModes2.py --file lightcurve.dat --p ini2.txt`
In this example a suffix "ini2" will be added to the output folder such as `\results\lightcurve_ini2` thus allowing to run the program with different configurations.

Since v0.1.1 MultiModes2 can be also imported from a Jupyter Notebook. For example:

```
>>> import MultiModes2 as mm2
>>> args = mm2.arguments(['--file', 'example.txt']
>>> mm2.multimodes(args)
```

The output, though, will not appear in the notebook yet.

## Notes
1) In clean_close only final results are cleaned and not intermediate files. Yake into account that the Rayleigh frequency is a limit to detect separate frequencies in the periodogram due to the leakage but this is different for a non-linear least squares in time, where the full information (not only the amplitudes) is used for the fitting and much closer frequencies can be fitted. Therefore, frequencies that are closer than Rayleigh in the final solution are not necessarily spurious and clean_close should be use with care, especially, if osratio is 1.

2) The results of frequency analyses are very dependent on the parameters used but, especially, the method used to estimate the noise level is very determining. Use the 'noise_method' parameter wisely.