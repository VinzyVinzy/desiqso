"""
This module contains functions to perform the cross-correlation analysis on the downloaded data respecting 
the selected mode. It loads the spectra from the local `.fits` files and performs the cross-correlation
analysis on them using parallel processing. The analysis is performed for all synthetic profiles, on a 
range of redshifts around the observed redshift of the quasar.
"""

# Importing necessary libraries
from astropy.convolution import convolve, Box1DKernel
from astropy.io import fits
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from scipy.stats import spearmanr
from tqdm import tqdm
import warnings

# Local imports
from src.desiqso.analysis.absorption_masks import compute_h2_absorption_masks
from src.desiqso.config import (SNR_THRESHOLD, VELOCITY_RANGE, NUM_REDSHIFT_VALUES, PLOT_CORRELATION_COEFFICIENTS, MULTIPLY_BY_CONTINUUM, PLOT_2D_DISTRIBUTION, SPECTRA_DATA_FOLDER, CROSS_CORRELATION_RESULTS_FOLDER)
from src.desiqso.constants import (H2_LYMAN_WERNER_BANDS, NUMBER_OF_BANDS, C_KMS, ColNames, Modes, PREL_LIST)
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.models.profile import (ProfileManager, Profile,)
from src.desiqso.models.results import CrossCorrelationResult
from src.desiqso.models.spectrum import SpectrumRecord
from src.desiqso.visualization.statistics import (plot_correlation_vs_redshift, plot_corrcoeff_vs_coretrans_2d)


# Parameter to set the number of cores to use for the cross-correlation analysis
NUMBER_OF_CORES = min(os.cpu_count(), 6)

# Function to run the cross-correlation analysis from the pipeline program
def run_cross_correlation_analysis(mode : str = Modes.ALL, spectra_files : list[str] = [], profiles_to_fit : list[Profile] = [], output_folder : str = CROSS_CORRELATION_RESULTS_FOLDER) -> None:
    """
    Function performing the cross-correlation analysis on the downloaded data respecting the selected 
    mode (see `Modes` enum). It loads the spectra
    from the local `.fits` files and performs the cross-correlation analysis on them using parallel
    processing. The results are saved in a `.txt` file, unique for each synthetic profiles. Also, 
    spectra with low SNR (< SNR_THRESHOLD) are not processed and are saved in the `low_snr.txt` file.

    :param MODE: Mode of selection. See `Modes` enum.
    :type MODE: str
    :param spectra_files: List of spectra files to process.
    :type spectra_files: list[str]
    :param profiles_to_fit: List of synthetic profiles to fit.
    :type profiles_to_fit: list[Profile]
    :param output_folder: Folder to save the results.
    :type output_folder: str
    :return None: This function does not return anything.
    """

    # ================
    # Configuration
    # ================

    # Creation of the results accumulator for the spectra analysis, to store the results of the cross-correlation analysis for all the loaded spectra
    results_accumulator = defaultdict(list)
    # If the list of spectra is empty
    if len(spectra_files) == 0:
        # Inform user
        print("\n[INFO] Loading spectra from local files...")
        # Retrieve the list of spectra files using the selected mode
        spectra_files = select_spectra_for_analysis(mode=mode)
        # Inform user
        print(f"[INFO] {len(spectra_files)} spectra loaded successfully!")
    # If the list of profiles to fit is empty
    if len(profiles_to_fit) == 0:
        # Load synthetic profiles
        ProfileManager.load_all()
        # Perform the analysis on all the available synthetic profiles
        profiles_to_fit = ProfileManager.all_profiles()

    # ================
    # Spectra loading and parallel processing
    # ================

    # Inform user
    print(f"\n[INFO] Processing spectra in parallel, using {NUMBER_OF_CORES} cores...")

    # Creation of Process Pool Executor for parallel processing
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), NUMBER_OF_CORES), initializer=init_worker) as executor:
        # Parallel processing of the cross-correlation analysis using the Process Pool Executor
        futures = [executor.submit(spectrum_analysis, spectrum_file, profiles_to_fit) for spectrum_file in spectra_files]
        # Loop on the futures to retrieve the results of the cross-correlation analysis
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing spectra", unit="file"):
            # Retrieve the results of the cross-correlation analysis
            result = future.result()
            # If the cross-correlation analysis failed
            if result is None:
                continue
            # If the cross-correlation analysis succeeded, retrieve the file name and the results
            file_name, results = result
            # Update the results accumulator using the file name and the results
            results_accumulator[file_name] = results

    # Inform user
    print(f"\n[INFO] Spectra processing completed successfully! {len(results_accumulator)} spectra processed in total.")
    
    # ================
    # Results saving
    # ================

    # Inform user
    print("\n[INFO] Saving spectra analysis results...")
    # Initialisation of results files
    if CrossCorrelationResult._results is None:
        CrossCorrelationResult.initialize_results(output_folder=output_folder, profiles_list=profiles_to_fit)
    # Save the results of the spectra analysis in the corresponding local files in .txt format
    for _, results in tqdm(results_accumulator.items(), desc="Saving results", unit="file"):
        results.save_results()
    # Inform user
    print(f"\n[INFO] Spectra analysis results saved successfully in folder `{output_folder}`!\n")

# Function to select only a small part of the downloaded spectra to perform the cross-correlation analysis
def select_spectra_for_analysis(mode:str = Modes.ALL) -> list :
    """
    Function to select a subset of the downloaded spectra to perform the cross-correlation analysis.

    - If mode is "all", it returns all the downloaded spectra.
    - If mode is "random", it returns a random sample of 20000 spectra from the downloaded spectra.
    - If mode is "preliminary", "confirmed", "borderline" or "rejected", it returns the corresponding respective data group from the preliminary analysis results.

    :param mode: Mode of selection. Can be "all", "random", "preliminary", "confirmed", "borderline" or 
    "rejected".
    :type mode: str
    
    :return: List of selected files.
    :rtype: list
    """
    
    # If the mode is "all"
    if mode == Modes.ALL:
        spectra_files = [file for file in os.listdir(SPECTRA_DATA_FOLDER) if file.endswith(".fits")]
    
    # If the mode is "random"
    elif mode == Modes.RANDOM:
        spectra_files = random.sample([file for file in os.listdir(SPECTRA_DATA_FOLDER) if file.endswith(".fits")], 10000)
    
    # If the mode is "preliminary", "confirmed", "borderline" or "rejected"
    elif mode in PREL_LIST:
        # Load preliminary analysis results
        AnalysisResults.load_preliminary_results()
        # Define data groups using preliminary analysis
        spectra_files   = set(AnalysisResults._preliminary_results[f"{mode}_candidates"][ColNames.FILENAME])
    
    elif mode == Modes.PRELIMINARY:
        # Load preliminary analysis results
        AnalysisResults.load_preliminary_results()
        # Define data groups using preliminary analysis
        spectra_files = set(AnalysisResults._preliminary_results["confirmed_candidates"][ColNames.FILENAME]) | set(AnalysisResults._preliminary_results["borderline_candidates"][ColNames.FILENAME]) | set(AnalysisResults._preliminary_results["rejected_candidates"][ColNames.FILENAME])

    elif mode == Modes.SAMPLE:
        # Load preliminary analysis results
        AnalysisResults.load_preliminary_results(verbose=False)
        # Initialize the list of selected files
        spectra_files = set()
        # Loop on the preliminary analysis categories
        for category in [Modes.CONFIRMED, Modes.REJECTED]:
            # Append the files of the current category to the list of selected files
            spectra_files = spectra_files | set(AnalysisResults._preliminary_results[f"{category}_candidates"][ColNames.FILENAME])
        # Append a random sample of 1000 files to the list with a fixed seed for reproducibility
        random.seed(0)
        spectra_files = spectra_files | set(random.sample([file for file in os.listdir(SPECTRA_DATA_FOLDER) if file.endswith(".fits")], 2000))
        # Convert the set to a list
        spectra_files = list(spectra_files)

    # Return the list of selected files
    return spectra_files

# Function to perform the cross-correlation analysis for a single spectrum using its filename
def spectrum_analysis(spectrum_file : str, profiles_to_fit : list[Profile]) -> tuple[str, CrossCorrelationResult]:
    """
    Function to perform the cross-correlation analysis for a single spectrum using its filename. It takes 
    as input the filename of a spectrum `.fits` file and returns a tuple containing the filename and 
    the results of the cross-correlation analysis as a `CrossCorrelationResult` instance. This function
    was designed to be used in parallel processing.

    :param spectrum_file: The filename of the spectrum `.fits` file
    :type spectrum_file: str
    :param profiles_to_fit: The list of profiles to fit
    :type profiles_to_fit: list[Profile]
    :return: A tuple containing the filename and the results of the cross-correlation analysis
    :rtype: tuple[str, CrossCorrelationResult]
    """

    # Try to perform the cross-correlation analysis
    try:
        if type(spectrum_file) == str:
            # Load the spectrum data from the current file using `astropy.io.fits`
            with fits.open(os.path.join(SPECTRA_DATA_FOLDER, spectrum_file), memmap=False) as hdul:
                # Convert the loaded spectrum data into a python object to easily manipulate it for the spectra analysis
                spectrum_record = SpectrumRecord.from_fits(hdul)
                # Using the dedicated method to perform spectrum analysis
                results = cross_correlate(spectrum_record, profiles_to_fit)
            # Return the results of the cross-correlation analysis
            return (results.file_name, results)
        else:
            results = cross_correlate(spectrum_file, profiles_to_fit)
            return (results.file_name, results)
    # If the cross-correlation analysis fails
    except Exception as e:
        # Inform user with an error
        tqdm.write(f"[ERROR] Cross-correlation analysis failed for spectrum `{spectrum_file}`: {e}")
        # Return None
        return None

# Function to perform cross-correlation analysis on a spectrum record
def cross_correlate(record : SpectrumRecord, profiles_to_fit : list[Profile]) -> CrossCorrelationResult:
    """
    Perform cross-correlation analysis between the observed flux and synthetic H₂ profiles.

    This method is used to detect the presence of molecular hydrogen (H₂) in a spectrum by
    cross-correlating the observed flux with synthetic H₂ profiles. The cross-correlation function
    is calculated for all synthetic H₂ profiles and the observed flux, for a range of redshifts to
    determine the best match.

    The method takes as argument a `SpectrumRecord` instance, and returns a `CrossCorrelationResult` 
    instance containing the status and results of the analysis. The cross-correlation values
    are calculated using the `scipy.stats.spearmanr` function.

    :param record: A `SpectrumRecord` instance containing the spectrum data.
    :type record: `SpectrumRecord`
    :param profiles_to_fit: A list of `Profile` instances to fit during the cross-correlation analysis.
    :type profiles_to_fit: list[Profile]
    :return: A `CrossCorrelationResult` instance containing the status and results of the analysis.
    :rtype: CrossCorrelationResult
    """
    
    # ==============
    # Pre-processing
    # ==============

    # Determining Lyman-Werner region for the spectrum using its redshift
    region = (record.wavelength >= H2_LYMAN_WERNER_BANDS[0]*(1.+record.redshift)) & (record.wavelength <= H2_LYMAN_WERNER_BANDS[1]*(1.+record.redshift))
    # Compute the constant continuum level in the Lyman-Werner region
    continuum = record.continuum
    # If the computed value is not valid
    if np.isnan(continuum):
        # Return a `CrossCorrelationResult` instance indicating that no valid continuum could be computed
        return CrossCorrelationResult(status        =   "Failed",
                                      file_name     =   record.filename,
                                      name          =   record.name,
                                      ra            =   record.ra,
                                      dec           =   record.dec,
                                      redshift      =   record.redshift,
                                      snr           =   record.snr,
                                      continuum     =   continuum,
                                      details       =   "Invalid continuum level")
    # Compute Continuum-to-Noise Ratio (CNR) in the Lyman-Werner region using the estimated 
    # continuum level and the error array
    CNR = continuum / record.err[region].mean()
    # Compute Signal-to-Noise Ratio (SNR) outside the Lyman-Werner region
    SNR = record.snr
    # Compute (if possible) signal to noise ration (SNR) in the SNR estimation region using 
    # the flux and error arrays
    if np.isnan(SNR) or SNR == 0.:
        # Return a `CrossCorrelationResult` instance indicating that no SNR could be computed
        return CrossCorrelationResult(status        =   "Failed",
                                      file_name     =   record.filename,
                                      name          =   record.name,
                                      ra            =   record.ra,
                                      dec           =   record.dec,
                                      redshift      =   record.redshift,
                                      snr           =   SNR,
                                      continuum     =   continuum,
                                      details       =   "No SNR could be computed")

    # ==============
    # Cross-correlation set-up
    # ==============

    # Required condition for the cross-correlation analysis
    if SNR < SNR_THRESHOLD:
        # Return a `CrossCorrelationResult` instance indicating that the spectrum has low SNR 
        # and the cross-correlation analysis was not performed if the required condition is not met
        return CrossCorrelationResult(status        =   "Low SNR", 
                                      file_name     =   record.filename, 
                                      name          =   record.name, 
                                      ra            =   record.ra, 
                                      dec           =   record.dec, 
                                      redshift      =   record.redshift, 
                                      snr           =   SNR,
                                      continuum     =   continuum)

    # Smooth the observed flux using a boxcar kernel to reduce noise and enhance the signal for the 
    # cross-correlation analysis
    flux_smoothed = convolve(record.flux, Box1DKernel(3))
    # Select the continuum level to use for the cross-correlation analysis based on the configuration
    continuum_level = continuum if MULTIPLY_BY_CONTINUUM else 1.
    # Compute the normalized smoothed flux using the continuum level and the mask
    n_flux_smoothed = flux_smoothed / continuum_level

    # Compute the redshift range to search for the H₂ absorption features based on the defined 
    # velocity range (see `VELOCITY_RANGE` constant), to find proximate absorption features that 
    # could be associated with the quasar host galaxy or its environment
    delta_z = (1 + record.redshift) * (VELOCITY_RANGE / C_KMS)
    # Compute the redshift values corresponding to the defined velocity range around the quasar 
    # redshift
    z_values = np.linspace(record.redshift - delta_z, record.redshift + delta_z, num=NUM_REDSHIFT_VALUES)
    # Remove redshift values outside the DESI data range
    z_values = z_values[z_values>2.4783]

    # Lists to store the results of the cross-correlation analysis for each synthetic H₂ profile
    best_redshifts                   = []
    best_correlation_coefficients    = []
    best_correlation_probabilities   = []
    best_core_transmission_values    = []
    best_J_core_transmission_values  = []
    best_correlation_parameters      = []

    # ==============
    # Cross-correlation analysis on the synthetic H₂ profiles
    # ==============

    # Perform cross-correlation analysis between the smoothed observed flux and the synthetic 
    # H₂ profiles for each redshift value in the defined range, to search for potential H₂ absorption 
    # features at different redshifts
    for profile in profiles_to_fit:

        # Initialize arrays to store the analysis results for the current synthetic H₂ profile
        correlation_coefficients   = np.full_like(z_values, np.nan)
        correlation_probabilities  = np.full_like(z_values, np.nan)
        core_transmissions_values  = np.full_like(z_values, np.nan)
        core_transmissions_levels  = []

        # Loop on redshift values to perform the cross-correlation analysis for each redshift values
        for i, z in enumerate(z_values):

            # Compute masks for H₂ absorption features and rebin synthetic profile using the 
            # current redshift, the synthetic H₂ profile and the dedicated function
            mask_data, mask_core, h2_synthetic_flux_rebinned = compute_h2_absorption_masks(record.wavelength, z, profile, record.mask)

            # If there are no pixels satisfying the selection criteria, skip the cross-correlation 
            # analysis for the current redshift value and move to the next one
            if not mask_data.any():
                continue

            # Retrieve the synthetic profile flux to fit
            flux_to_fit = h2_synthetic_flux_rebinned[mask_data]

            # If the flux to fit is constant, skip the cross-correlation analysis for the current 
            # redshift value and move to the next one
            if np.std(flux_to_fit) == 0 or np.std(n_flux_smoothed[mask_data]) == 0:
                # Warn the user, specifying the spectrum name and redshift value, as well as the flux to fit and the normalized smoothed flux to fit
                warnings.warn(f"For spectrum {record.name} at redshift {z}, the synthetic profile flux to fit or the normalized smoothed flux to fit is constant.", RuntimeWarning)
                # Continue to the next redshift value
                continue

            # Compute the Spearman rank correlation coefficient and correlation probability
            correlation_coefficient, correlation_probability = spearmanr(flux_to_fit, n_flux_smoothed[mask_data])

            # Saving the results of the cross-correlation analysis for the current redshift value, if the results are better than the previous ones
            correlation_coefficients[i]  = correlation_coefficient
            correlation_probabilities[i] = correlation_probability

            # ==============
            # Core transmission computation
            # ==============

            # If there are any pixels in the mask
            if mask_core.any():
                # Defining the core transmission array using the normalized smoothed flux and the rebinned synthetic profile flux
                core_transmission_array = (n_flux_smoothed - h2_synthetic_flux_rebinned)[mask_core]
                # Compute the median transmission in the core of H₂ absorption features to identify strong absorber
                core_transmissions_values[i] = np.median(core_transmission_array)
                
                # Check that the mask is in bool type
                mask = mask_core.astype(bool)
                # Find peaks
                diff = np.diff(mask.astype(int))
                # Update the start and end indices of the segments where the mask is True
                starts = np.where(diff == 1)[0] + 1
                ends   = np.where(diff == -1)[0] + 1
                # Manage the limits of the masks
                if mask[0]:
                    starts = np.insert(starts, 0, 0)
                if mask[-1]:
                    ends = np.append(ends, len(mask))
                
                # If the mask doesn't have exactly 6 components, fuse neighboring components
                if len(starts) != NUMBER_OF_BANDS and len(starts) > 0 and len(ends > 0):
                    # Ensure that the starts and ends arrays have the same length
                    n = min(len(starts), len(ends))
                    starts = starts[:n]
                    ends   = ends[:n]
                    # Building segments
                    segments = list(zip(starts, ends))
                    # Initialize a list containing the merged segments
                    merged_segments = [segments[0]]
                    # Loop on the segments
                    for start, end in segments[1:]:
                        # Retrieve the last merged segment
                        prev_start, prev_end = merged_segments[-1]
                        # If the current segment overlaps with the previous one, merge them
                        if start - prev_end <= 15:
                            # Update the merged segment list
                            merged_segments[-1] = (prev_start, end)
                        # If the current segment doesn't overlap with the previous one, add it to the merged segment list
                        else:
                            # Add it to the merged segment list
                            merged_segments.append((start, end))
                    # Convert the merged segments to arrays
                    starts = np.array([start for start, _ in merged_segments])
                    ends   = np.array([end for  _, end in merged_segments])

                # If after merging the neighboring components, the mask still doesn't have exactly 6 components, save NaN values for the core transmissions and continue to the next redshift value
                if len(starts) != NUMBER_OF_BANDS:
                    core_transmissions_levels.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                    continue

                # Retrieve the core transmissions in each line and save them
                lines_core_transmissions = [np.median((n_flux_smoothed - h2_synthetic_flux_rebinned)[start:end]) for start, end in zip(starts, ends)]
                core_transmissions_levels.append(lines_core_transmissions)
        
        # ==============
        # Plotting (if required)
        # ==============        
            
        # If required, call the function to plot the corelation analysis as a function of redshift
        if PLOT_CORRELATION_COEFFICIENTS:
            plot_correlation_vs_redshift(z_values, correlation_coefficients, core_transmissions_values, core_transmissions_levels, record.filename, profile, record.redshift, delta_z/NUM_REDSHIFT_VALUES, SNR)
        
        # If required, call the function to plot the 2D distribution of correlation coefficients and core transmissions
        if PLOT_2D_DISTRIBUTION:
            plot_corrcoeff_vs_coretrans_2d(z_values, correlation_coefficients, core_transmissions_values, record.filename, profile, record.redshift, delta_z/NUM_REDSHIFT_VALUES, SNR)
        
        # ==============
        # Processing results
        # ==============

        # If there are no valid correlation coefficients, set the best correlation index to None
        if np.all(np.isnan(correlation_coefficients)):
            # Return a `CrossCorrelationResult` instance indicating that the spectrum analysis was not successful
            return CrossCorrelationResult(status        =   "Failed", 
                                          file_name     =   record.filename, 
                                          name          =   record.name, 
                                          ra            =   record.ra, 
                                          dec           =   record.dec, 
                                          redshift      =   record.redshift, 
                                          snr           =   SNR,
                                          continuum     =   continuum,
                                          details       =   "No valid correlation coefficients")  
        # Find the best correlation index in all the results
        else:
            correlation_parameters = correlation_coefficients*(1-core_transmissions_values)
            max_correlation_index = np.nanargmax(correlation_parameters)
            # Append the best correlation results for the current synthetic H₂ profile to the corresponding lists
            best_redshifts.append(z_values[max_correlation_index])
            best_correlation_coefficients.append(correlation_coefficients[max_correlation_index])
            best_correlation_probabilities.append(correlation_probabilities[max_correlation_index])
            best_core_transmission_values.append(core_transmissions_values[max_correlation_index])
            best_J_core_transmission_values.append(core_transmissions_levels[max_correlation_index])
            best_correlation_parameters.append(correlation_parameters[max_correlation_index])
    
    # Check to see if all J core transmissions are valid
    if any(np.all(np.isnan(sublist)) for sublist in best_J_core_transmission_values):
        # Return a `CrossCorrelationResult` instance witth the "Failed" status
        return CrossCorrelationResult(status    =   "Failed", 
                                      file_name   =   record.filename, 
                                      name        =   record.name,
                                      ra          =   record.ra,
                                      dec         =   record.dec,
                                      redshift    =   record.redshift, 
                                      snr         =   SNR,
                                      continuum   =   continuum,
                                      details     =   "No valid core transmissions") 
    
    # Create list to find best fit
    best_fit_flags = [1 if param == max(best_correlation_parameters) else 0 for param in best_correlation_parameters]

    # Return a `CrossCorrelationResult` instance containing the results of the cross-correlation 
    # analysis for the current spectrum record
    return CrossCorrelationResult(status                     =   "Success",
                                  file_name                  =   record.filename,
                                  name                       =   record.name,
                                  ra                         =   record.ra,
                                  dec                        =   record.dec,
                                  redshift                   =   record.redshift,
                                  best_redshifts             =   best_redshifts,
                                  best_correlation_values    =   best_correlation_coefficients,
                                  best_correlation_probs     =   best_correlation_probabilities,
                                  best_core_transmissions    =   best_core_transmission_values,
                                  best_J_core_transmissions  =   best_J_core_transmission_values,
                                  best_correlation_parameters=   best_correlation_parameters,
                                  snr                        =   SNR,
                                  continuum                  =   continuum,
                                  best_fit_flags             =   best_fit_flags)

# Function to initialize the worker for the parallel processing of the spectra analysis, by loading the H₂ synthetic profiles in memory
def init_worker():
    """
    Initializes the worker for the parallel processing of the spectra analysis by turning off interactive 
    plots to prevent visual artifacts.

    This function is intended to be used as an initializer for a `ProcessPoolExecutor`, and should not be 
    called directly.
    """
    
    
    # Turn off interactive plots to prevent visual artifacts
    plt.ioff()
