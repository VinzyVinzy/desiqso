"""
This module contains functions to create a sample of mock spectra and perform the cross-correlation analysis on it.
It also plots some of the statistics and spectra from the mock spectra analysis results.
"""

# Packages import
from collections import defaultdict
from concurrent.futures import (ProcessPoolExecutor, as_completed,)
from math import log10
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm

# Local imports
from src.desiqso.analysis.absorption_masks import compute_h2_absorption_masks
from src.desiqso.analysis.cross_correlation import (NUMBER_OF_CORES, select_spectra_for_analysis, spectrum_analysis, init_worker,)
from src.desiqso.config import (MOCK_ANALYSIS_FOLDER, CORRELATION_PARAM_THRESHOLD,)
from src.desiqso.constants import (ColNames, Modes, Categories,)
from src.desiqso.data.loader import load_spectrum_from_filename
from src.desiqso.models.dataset import (AnalysisResults, which_data_group,)
from src.desiqso.models.profile import ProfileManager
from src.desiqso.models.results import CrossCorrelationResult
from src.desiqso.models.spectrum import SpectrumRecord
from src.desiqso.utils.helpers import (parse_cell, compute_grade, compute_relative_speed, _is_valid,)
from src.desiqso.visualization.spectra import plot_spectrum
from src.desiqso.visualization.statistics import plot_distribution

# Function to perform the cross-correlation analysis on a sample of mock spectra
def mock_analysis(profile_ntot : float = 20., use_same_profile : bool = True) -> dict[str, SpectrumRecord]:
    """
    This function performs the cross-correlation analysis on a sample of mock spectra.
    The mock spectra are generated based on high-SNR (> 15) spectra with low-chance of H₂ 
    multiplied by a complete H₂ profile at the redshift of the quasar with added noise.
    If this analysis is already performed, the function loads the results from the output folder.

    :param profile_ntot: The total column density value of the profile to use for the mock spectra. Default is 20.
    :type profile_ntot: float, optional
    :param use_same_profile: Whether to use the same profile for all the mock spectra. Default is True.
    :type use_same_profile: bool, optional
    :return dict[str, SpectrumRecord]: A dictionary of the mock spectra with the key being the filename and the value being 
    the associated `SpectrumRecord` instance.
    """

    # ====================
    # Base configuration
    # ====================

    # Inform user
    print("\n[INFO] Starting mock spectra analysis...")

    # Creating outputfolder if it doesn't exist
    os.makedirs(MOCK_ANALYSIS_FOLDER, exist_ok=True)

    # Output file path
    output_file = os.path.join(MOCK_ANALYSIS_FOLDER, "mock_spectra.pk1")

    # If the output folder is not empty
    if len(os.listdir(MOCK_ANALYSIS_FOLDER)) > 0:
        # Reading the dictionary of the mock spectra
        with open(output_file, "rb") as file:
            mock_spectra = pickle.load(file)
        # Inform user
        print(f"[INFO] {len(mock_spectra)} mock spectra loaded successfully!")
        # Returning the dictionary
        return mock_spectra
    
    # ====================
    # Sample selection
    # ====================

    # Inform user
    print("[INFO] Selecting sample...")

    # Retrieve test sample
    sample_spectra = select_spectra_for_analysis(mode=Modes.SAMPLE)
    
    # Loading the cross-correlation analysis results
    AnalysisResults.load_results(verbose=False)
    # Retrieving the table corresponding to the spectra with high SNR and with low chances of having H₂
    table = AnalysisResults.results_survey(mode=Modes.ALL, profile_name="best", thresholds_dict={ColNames.SNR : (15, None), ColNames.CORR_PARAM : (None, 0.3)})
    # Selecting only spectra that are not from the "confirmed", "borderline" or "rejected" categories
    table = table[table[ColNames.CATEGORY] == Categories.OTHER]
    # Selecting spectra that are not already in the sample
    table = table[~table[ColNames.FILENAME].isin(sample_spectra)]

    # Keeping only a sample of a 1000 spectra filenames with a fixed seed for reproducibility
    filenames = table[ColNames.FILENAME].sample(n=1000, random_state=0).tolist()
    # Adding the other spectra to the sample
    sample_spectra.extend(table[~table[ColNames.FILENAME].isin(filenames)][ColNames.FILENAME].tolist())

    # Loading synthetic H₂ profiles
    ProfileManager.load_all(verbose=False)
    # If there is more than one profile
    if len(ProfileManager.all_profiles()) > 1:
        # Retrieving the first profile with the selected Ntot
        profile = [profile for profile in ProfileManager.all_profiles() if log10(profile.Ntot) == profile_ntot][0]
    # If there is only one profile
    else :
        profile = ProfileManager.all_profiles()[0]
    # Retrieving the complete version of the synthetic profile
    full_profile = profile.get_complete_profile()
    
    # If the user does not want to use the same profile for the analysis, removing the profile from the manager
    if not use_same_profile and len(ProfileManager.all_profiles()) > 1:
        ProfileManager._profiles.pop(profile.name)

    # ====================
    # Creating and saving the mock spectra
    # ====================

    # Initialize dictionary containing the filenames and corresponding records of the mock spectra
    mock_spectra = {}

    # Looping on the filenames
    for filename in tqdm(filenames, desc="Creating mock spectra...", unit="spectra"):

        # Opening the `.fits` file corresponding as a `SpectrumRecord` instance
        record = load_spectrum_from_filename(filename)
        # Normalizing the flux by the continuum
        record.flux /= record.continuum
        # Rebinning the synthetic profile to match the record
        mask_data, _, flux_rebinned = compute_h2_absorption_masks(record.wavelength, record.redshift, full_profile, record.mask)
        # Find the indices of the data points that are not masked
        indices = np.where(mask_data)[0]
        # Creating the slice for the synthetic profile
        indices_slice = slice(indices[0]-10, indices[-1]+10)
        # Adding the synthetic profile to the original spectrum
        record.flux[indices_slice] *= flux_rebinned[indices_slice]
        # Selecting a random SNR
        snr = float(np.random.randint(3, 10))
        # Computing the noise
        sigma = 1/snr
        noise = np.random.normal(0, sigma, len(record.wavelength))
        # Adding the noise to the flux
        record.flux += noise
        # Updating the error
        record.err = np.full_like(record.wavelength, sigma)
        # Adding the record to the sample list
        sample_spectra.append(record)
        # Adding the record to the dictionary
        mock_spectra[filename] = record
    
    # Saving the dictionary of the mock spectra in a pickle file
    with open(output_file, "wb") as file:
        pickle.dump(mock_spectra, file)

    # ====================
    # Cross-correlation analysis
    # ====================
    
    # Creation of the results accumulator for the spectra analysis, to store the results of the cross-correlation analysis for all the loaded spectra
    results_accumulator = defaultdict(list)
    
    # Inform user
    print(f"\n[INFO] Processing mock spectra in parallel, using {NUMBER_OF_CORES} cores...")

    # Creation of Process Pool Executor for parallel processing
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), NUMBER_OF_CORES), initializer=init_worker) as executor:
        # Parallel processing of the cross-correlation analysis using the Process Pool Executor
        futures = [executor.submit(spectrum_analysis, spectrum_file) for spectrum_file in sample_spectra]
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
        CrossCorrelationResult.initialize_results(output_folder=MOCK_ANALYSIS_FOLDER)
    # Save the results of the spectra analysis in the corresponding local files in .txt format
    for _, results in tqdm(results_accumulator.items(), desc="Saving results", unit="file"):
        results.save_results()
    # Inform user
    print(f"\n[INFO] Spectra analysis results saved successfully in folder `{MOCK_ANALYSIS_FOLDER}`!")

    # Return the list of mock spectra filenames
    return mock_spectra

# Function to plot some of the statistics and spectra from the mock spectra analysis results
def mock_spectra_statistics_plotting(mock_spectra : dict[str, SpectrumRecord], profile_ntot : float = 20.) -> None :
    """
    This function plots some of the statistics and spectra from the mock spectra analysis results.

    :param mock_spectra: Dictionary containing the filenames and corresponding `SpectrumRecord` of the mock spectra.
    :type mock_spectra: dict[str, SpectrumRecord]
    :param profile_ntot: Total column density of the synthetic profile used for creating the mock spectra.
    :type profile_ntot: float
    :return: This function does not return anything.
    :rtype: None
    """

    # ================
    # Results loading
    # ================

    # Inform user
    print("\n[INFO] Loading spectra analysis results...")

    # Preliminary analysis results loading
    AnalysisResults.load_preliminary_results(verbose=False)
    # Loading all the H₂ synthetic profiles
    ProfileManager.load_all(verbose=False)
    # If there is more than one profile
    if len(ProfileManager.all_profiles()) > 1:
        # Retrieving the first profile with the selected Ntot
        profile = [profile for profile in ProfileManager.all_profiles() if log10(profile.Ntot) != profile_ntot][0]
    # If there is only one profile
    else :
        profile = ProfileManager.all_profiles()[0]

    # Reading the results of the analysis
    result_file = [file for file in os.listdir(MOCK_ANALYSIS_FOLDER) if file.endswith(".npy")][0]
    results = pd.read_csv(MOCK_ANALYSIS_FOLDER+result_file, sep="\t")
    # Cleaning the column names
    results.columns = results.columns.str.replace("# ", "", regex=False)
    # Parsing the cells to obtain the right data type for the column "Best fit J core transmission"
    if ColNames.J_CORE_TRANS in results.columns:
        # Applying the `parse_cell` function to convert the cell to a list
        results[ColNames.J_CORE_TRANS] = results[ColNames.J_CORE_TRANS].apply(parse_cell)
        # Applying the `compute_grade` function to obtain the grade directly in the DataFrame
        results[ColNames.GRADE] = results[ColNames.J_CORE_TRANS].apply(compute_grade)
        # Applying the `_is_valid` function to obtain the validity directly in the DataFrame
        results[ColNames.IS_VALID] = results.apply(lambda row: _is_valid(row), axis=1)
    # Applying the `which_data_group` function to obtain the data group directly in the DataFrame
    results[ColNames.CATEGORY] = results[ColNames.FILENAME].apply(which_data_group)
    # Adding the name of the synthetic profile
    results[ColNames.PROFILE] = result_file[:-4]
    # Applying the `compute_relative_speed` function to obtain the relative speed directly in the DataFrame
    if ColNames.Z in results.columns:
        # Applying the `compute_relative_speed` function to obtain the relative speed directly in the DataFrame
        results[ColNames.REL_SPEED] = results.apply(lambda row: compute_relative_speed(row[ColNames.Z], row[ColNames.QSO_Z]), axis=1)
    
    # ================
    # Plotting statistics distributions
    # ================

    # Inform user
    print("\n[INFO] Plotting statistics distributions...\n")

    # Defining the pairs of statistics to plot
    plot_pairs = [
        (ColNames.SNR, ColNames.CORR_PARAM),
        (ColNames.CORR_PARAM, ColNames.CORE_TRANS),
        (ColNames.CORR_COEFF, ColNames.CORE_TRANS),
        (ColNames.Z, ColNames.CORR_PARAM),
    ]

    # Table containing the results for the spectra in which H2 was added
    mock_results = results.copy()
    mock_results = mock_results[mock_results[ColNames.FILENAME].isin(mock_spectra.keys())]
    # 
    mock_results = mock_results[mock_results[ColNames.PROFILE] == profile.name]
    # Table containing the results for the spectra in which H2 was added and the cross-correlation analysis succeeded
    mock_results_valid = mock_results[mock_results[ColNames.CORR_PARAM] >= CORRELATION_PARAM_THRESHOLD].copy()
    # Table containing the results for the spectra in which H2 was added and the cross-correlation analysis succeeded
    mock_results_not_valid = mock_results[mock_results[ColNames.CORR_PARAM] < CORRELATION_PARAM_THRESHOLD].copy()

    # Defining the data, savepath and label to plot
    plot_params = [
        (results, f"{MOCK_ANALYSIS_FOLDER}figures/statistics_all/", "MOCK ANALYSIS : All spectra"),
        (mock_results, f"{MOCK_ANALYSIS_FOLDER}figures/statistics_mock/", "MOCK ANALYSIS : Mock spectra"),
        (mock_results_valid, f"{MOCK_ANALYSIS_FOLDER}figures/statistics_mock_valid/", "MOCK ANALYSIS : Mock spectra (valid)"),
        (mock_results_not_valid, f"{MOCK_ANALYSIS_FOLDER}figures/statistics_mock_not_valid/", "MOCK ANALYSIS : Mock spectra (not valid)"),
    ]

    # Looping over the plot parameters
    for results, savepath, label in plot_params:
        # If the results are empty or less than 10, continue
        if len(results) < 5:
            continue
        # Calling the `plot_distribution` function to plot the statistics
        plot_distribution(plot_pairs=plot_pairs, thresholds={}, profile_name="", mode=Modes.ALL, data=results, savepath=savepath, add_label=label)

    # ================
    # Plotting mock spectra
    # ================

    # Selecting 50 random valid mock spectra with a fixed seed for reproductibility
    table = mock_results_valid.sample(n=min(20, len(mock_results_valid)), random_state=42)
    # Loop on the list of valid spectra
    for _, row in tqdm(table.iterrows(), total=len(table), desc="Plotting valid mock spectra", unit="spectra"):
        # Plotting the spectrum using the dedicated function, without showing the plot
        plot_spectrum(row=row, folderpath="", record=mock_spectra[row[ColNames.FILENAME]], output_folder=f"{MOCK_ANALYSIS_FOLDER}figures/spectra_mock_valid/")
    
    # Selecting 50 random not valid mock spectra with a fixed seed for reproductibility
    table = mock_results_not_valid.sample(n=min(20, len(mock_results_not_valid)), random_state=42)
    # Loop on the list of valid spectra
    for _, row in tqdm(table.iterrows(), total=len(table), desc="Plotting not valid mock spectra", unit="spectra"):
        # Plotting the spectrum using the dedicated function, without showing the plot
        plot_spectrum(row=row, folderpath="", record=mock_spectra[row[ColNames.FILENAME]], output_folder=f"{MOCK_ANALYSIS_FOLDER}figures/spectra_mock_not_valid/")

    # Returning to main programm
    return
