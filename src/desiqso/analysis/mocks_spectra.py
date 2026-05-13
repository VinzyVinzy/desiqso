"""
This module contains functions to create a sample of mock spectra and perform the cross-correlation analysis on it.
It also plots some of the statistics and spectra from the mock spectra analysis results.
"""

# Packages import
from math import log10
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import pickle
from tqdm import tqdm

# Local imports
from src.desiqso.analysis.absorption_masks import compute_h2_absorption_masks
from src.desiqso.analysis.cross_correlation import (run_cross_correlation_analysis, select_spectra_for_analysis)
from src.desiqso.config import (settings, RESULTS_FOLDER,CORRELATION_PARAM_THRESHOLD,)
from src.desiqso.constants import (ColNames, COLUMN_FILE_LABELS, Modes, Categories,)
from src.desiqso.data.loader import load_spectrum_from_filename
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.models.profile import Profile
from src.desiqso.models.spectrum import SpectrumRecord
from src.desiqso.visualization.spectra import plot_spectrum
from src.desiqso.visualization.statistics import plot_distribution


# Function to create the mock spectra from a sample of high-SNR (> 15) spectra with low-chance of H₂
def create_mock_spectra(profile_to_add : Profile, output_file : str) -> tuple[dict[str, SpectrumRecord], list[str]]:
    """
    This function creates a sample of mock spectra from a sample of high-SNR (> 15) spectra with low-chance of H₂.
    It returns a dictionary containing the filenames and corresponding modified records of the mock spectra as well
    as the list of spectra to perform the cross-correlation analysis on.

    :param profile_to_add: The synthetic H₂ profile to add to the mock spectra.
    :type profile_to_add: Profile
    :param output_file: The output file to save the mock spectra.
    :type output_file: str

    :return tuple[dict[str, SpectrumRecord], list[str]]: A tuple containing the dictionary containing the filenames 
    and corresponding modified records of the mock spectra and the list of spectra to perform the cross-correlation analysis on.
    """

    # ====================
    # Sample selection
    # ====================

    # Inform user
    print("[INFO] Selecting sample for mock spectra creation...")

    # Retrieve test sample
    sample_spectra = select_spectra_for_analysis(mode=Modes.SAMPLE)
    
    # Loading the cross-correlation analysis results
    AnalysisResults.load_results(verbose=False)
    # Retrieving the table corresponding to the spectra with high SNR and with low chances of having H₂
    table = AnalysisResults.results_survey(mode=Modes.ALL, profile_name="best", thresholds_dict={ColNames.SNR : (15, None), ColNames.CORR_PARAM : (None, 0.3)})
    # Selecting only spectra that are not from the "confirmed", "unsure" or "rejected" categories
    table = table[table[ColNames.CATEGORY] == Categories.OTHER]
    # Selecting spectra that are not already in the sample
    table = table[~table[ColNames.FILENAME].isin(sample_spectra)]

    # Keeping only a sample of a 1000 spectra filenames with a fixed seed for reproducibility
    filenames = table[ColNames.FILENAME].sample(n=1000, random_state=0).tolist()
    # Adding the other spectra to the sample
    sample_spectra.extend(table[~table[ColNames.FILENAME].isin(filenames)][ColNames.FILENAME].tolist())
    
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
        mask_data, _, flux_rebinned = compute_h2_absorption_masks(record.wavelength, record.redshift, profile_to_add, record.mask)
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
    
    # Returning the dictionary of the mock spectra and the list of sample spectra to perform the cross-correlation analysis on
    return mock_spectra, sample_spectra

# Function to plot some of the statistics of the mock spectra analysis as well as some spectra
def mock_spectra_statistics(mock_spectra : dict[str, SpectrumRecord], profile_to_fit : Profile, output_folder = str) -> None:
    """
    This function performs the statistical analysis of the results from the cross-correlation analysis 
    on the sample containing the mock spectra. It plots some statistics distributions for multiple 
    data groups and a few mock spectra to manually inspect the results.
    
    :param mock_spectra: A dictionary containing the filenames and corresponding records of the mock spectra.
    :type mock_spectra: dict[str, SpectrumRecord]
    :param profile_to_fit: The profile the corss-correlation analysis was performed with.
    :type profile_to_fit: Profile
    :param output_folder: The folder where to save the plots.
    :type output_folder: str
    """

    # ================
    # Plotting mock spectra statistics
    # ================

    # Inform user
    print("\n[INFO] Plotting statistics distributions...")

    # Turn off interactive mode to prevent displaying bugs
    plt.ioff()
    # Update plot settings
    settings["ytick.right"] = True
    settings["xtick.top"]   = True
    plt.rcParams.update(**settings)

    # Defining the pairs of statistics to plot
    plot_pairs = [
#        (ColNames.SNR, ColNames.CORR_PARAM),
#        (ColNames.CORR_PARAM, ColNames.CORE_TRANS),
        (ColNames.CORR_COEFF, ColNames.CORE_TRANS),
#        (ColNames.Z, ColNames.CORR_PARAM),
#        (ColNames.GRADE, ColNames.CORR_PARAM),
    ]

    # Retrieve the results table from the `AnalysisResults` class
    results = AnalysisResults._results.copy()
    # Retrieve the table containing the results for the spectra in which H₂ was added and the analysis was a success
    mock_results = results[results[ColNames.FILENAME].isin(mock_spectra.keys())].copy()
    # Retrieve the table containing the results for the mock spectra successfully found
    mock_success = mock_results[mock_results[ColNames.IS_VALID] == 1].copy()
    # Retrieve the table containing the results for the mock spectra not found
    mock_failed = mock_results[mock_results[ColNames.IS_VALID] == 0].copy()
    # Retrieve the table containing the results for the false mock spectra
    false_mock = results[~results[ColNames.FILENAME].isin(mock_spectra.keys())].copy()

    # Defining the data, save path and additionnal labels to plot on the statistics plots
    plot_params = [
        (results,      f"{output_folder}stats/statistics_all/",           "MOCK ANALYSIS : All spectra"),
        (mock_results, f"{output_folder}stats/statistics_mock/",          "MOCK ANALYSIS : Mock spectra"),
        (mock_success, f"{output_folder}stats/statistics_mock_valid/",    "MOCK ANALYSIS : Mock spectra (valid)"),
        (mock_failed,  f"{output_folder}stats/statistics_mock_not_valid/","MOCK ANALYSIS : Mock spectra (not valid)"),
        (false_mock,   f"{output_folder}stats/statistics_false_mock/",    "MOCK ANALYSIS : False mock spectra"),
    ]

    # Looping over the plot parameters
    for table, savepath, label in plot_params:
        # If the results are empty or less than 10, continue
        if len(table) < 5:
            continue
        # Calling the `plot_distribution` function to plot the statistics distributions
        plot_distribution(plot_pairs=plot_pairs, thresholds={}, profile_name=profile_to_fit.name, color_col=ColNames.SNR, mode=Modes.ALL, data=table, savepath=savepath, add_label=label)
    
    # ================
    # Plotting mock spectra
    # ================

    # Inform user
    print("\n[INFO] Plotting mock spectra...")

    # Selecting maximum 10 random valid mock spectra with a fixed seed for reproductibility
    table = mock_success.sample(n=min(5, len(mock_success)), random_state=42)
    # Loop on the list of valid spectra
    for _, row in tqdm(table.iterrows(), total=len(table), desc="Plotting valid mock spectra", unit="spectra"):
        # Plotting the spectrum using the dedicated function, without showing the plot
        plot_spectrum(row=row, folderpath="", record=mock_spectra[row[ColNames.FILENAME]], output_folder=f"{output_folder}spectra_mock_valid/")
    
    # Selecting maximum 10 random not valid mock spectra with a fixed seed for reproductibility
    table = mock_failed.sample(n=min(5, len(mock_failed)), random_state=42)
    # Loop on the list of valid spectra
    for _, row in tqdm(table.iterrows(), total=len(table), desc="Plotting not valid mock spectra", unit="spectra"):
        # Plotting the spectrum using the dedicated function, without showing the plot
        plot_spectrum(row=row, folderpath="", record=mock_spectra[row[ColNames.FILENAME]], output_folder=f"{output_folder}spectra_mock_not_valid/")

    # Returning to the main function
    return

# Function to study the evolution of sample completeness with the threshold on given columns
def sample_completeness_analysis(mock_spectra : dict[str, SpectrumRecord],  output_folder : str, profile_added : Profile, profile_fitted : Profile, columns : list[str] = [ColNames.CORR_PARAM]) -> None:
    """
    This function plots the evolution of the valid sample completeness and purity with the threshold on given columns.

    :param mock_spectra: Dictionary of mock spectra
    :type mock_spectra: dict[str, SpectrumRecord]
    :param output_folder: Output folder path
    :type output_folder: str
    :param profile_added: Profile added to the mock spectra
    :type profile_added: Profile
    :param profile_fitted: Profile fitted during the cross-correlation analysis
    :type profile_fitted: Profile
    :param columns: List of columns to threshold on, defaults to [ColNames.CORR_PARAM]
    :type columns: list[str], optional
    """

    # ================
    # Configuration
    # ================

    # Inform user
    print("\n[INFO] Plotting sample completeness analysis...")

    # Making sure the output folder exists
    output_folder = f"{output_folder}sample_completeness/"
    os.makedirs(output_folder, exist_ok=True)

    # Turn off interactive mode to prevent displaying bugs
    plt.ioff()

    # Update plot settings
    settings["ytick.right"] = True
    settings["xtick.top"]   = True
    plt.rcParams.update(**settings)

    # Retrieve the mock spectra analysis results
    results = AnalysisResults._results.copy()
    # Mask to select only the spectra in which there is H₂ to find
    mask = (results[ColNames.FILENAME].isin(mock_spectra.keys())) | (results[ColNames.CATEGORY] == Categories.CONFIRMED)
    # Table containing the results for the spectra in which there is H₂ to find
    results_true = results[mask].copy()
    # Table containing the results for the spectra in which there is no H₂ to find
    results_false = results[~mask].copy() 

    # Defining the dictionnary containing the thresholds to loop over
    thresholds_dict = {
        ColNames.CORR_PARAM : np.linspace(0.0, 1.0, 100),
        ColNames.CORR_COEFF : np.linspace(0.0, 1.0, 100),
        ColNames.CORE_TRANS : np.linspace(-1.0, 1.0, 200),
        ColNames.GRADE      : np.linspace(0.0, 6.0, 7),
    }

    # ================
    # Completness and purity analysis
    # ================
    
    # Looping over the columns
    for column in tqdm(columns, desc="Completeness analysis", unit="columns"):

        # Inform user
        tqdm.write(f"\n[INFO] Plotting {column.lower()} for mock spectra sample...")

        # Retrieve the thresholds for the current column
        thresholds = thresholds_dict[column]

        # Selecting comparator based on the selected column
        comparator = operator.le if column in [ColNames.CORE_TRANS] else operator.ge

        # Initializing the arrays to store the completenesses and purities
        completenesses = np.zeros(len(thresholds))
        purities = np.zeros(len(thresholds))
        false_detections = np.zeros(len(thresholds))

        # Looping over some values to set as thresholds for the current column
        for i, threshold in enumerate(thresholds):

            # Compute the completeness for the current threshold and add it to the array
            completeness = len(results_true[comparator(results_true[column], threshold)]) / len(results_true) if len(results_true) > 0 else 1
            completenesses[i] = completeness*100
            # Compute the purity for the current threshold and add it to the array
            purity = len(results_true[comparator(results_true[column], threshold)]) / (len(results_true[comparator(results_true[column], threshold)]) + len(results_false[comparator(results_false[column], threshold)])) if (len(results_true[comparator(results_true[column], threshold)]) + len(results_false[comparator(results_false[column], threshold)])) else np.nan
            purities[i] = purity*100
            # Compute the number of false detections for the current threshold and add it to the array
            false_detection = len(results_false[comparator(results_false[column], threshold)]) / len(results_false) if len(results_false) > 0 else 1
            false_detections[i] = false_detection*100

        # Creating the figure and axis
        _, ax = plt.subplots(figsize=(14,8))
        # Plotting the completenesses
        ax.plot(thresholds, completenesses, color='blue', label=rf"Completeness ($C$)")
        # Plotting the purities
        ax.plot(thresholds, purities, color='green', label=rf"Purity ($P$)")
        # Plotting the false detections
        ax.plot(thresholds, false_detections, color='red', label=rf"False detection rate ($FR$)")
        # Displaying the number of spectra used
        ax.text(0.05, 0.10, rf"N$_{{H_2}}$ = {len(results_true)}"+"\n"+rf"N$_{{~H_2}}$ = {len(results_false)}", transform=ax.transAxes, fontsize=12, va='top')
        # Finding the best threshold, corresponding to the first threshold value for which the false detection rate is inferior to 1%
        best_threshold = thresholds[np.argmax(false_detections < 1)]
        plt.axvline(best_threshold, color='red', linestyle='--', linewidth=1.5, label=rf"Best threshold: {best_threshold:.2f}, $C=${completenesses[np.argmax(false_detections < 1)]:.2f}%, $P=${purities[np.argmax(false_detections < 1)]:.2f}%, $FR=${false_detections[np.argmax(false_detections < 1)]:.2f}%")
        # Displaying the current threshold and the associated completeness
        if column == ColNames.CORR_PARAM:
            plt.axvline(CORRELATION_PARAM_THRESHOLD, color="k", linestyle=":", linewidth=1.5, label=f"Current threshold: {CORRELATION_PARAM_THRESHOLD:.2f}, $C=${completenesses[np.argmin(np.abs(thresholds - CORRELATION_PARAM_THRESHOLD))]:.2f}%, $P=${purities[np.argmin(np.abs(thresholds - CORRELATION_PARAM_THRESHOLD))]:.2f}%, $FR=${false_detections[np.argmin(np.abs(thresholds - CORRELATION_PARAM_THRESHOLD))]:.2f}%")
        # Setting x-axis and y-axis limits
        plt.xlim(min(thresholds), max(thresholds))
        plt.ylim(-5, 105)
        # Adding the legend and title
        plt.title(f"Evolution of algorithm completeness, purity and false detection rate with the threshold on {column}\nFitted profile: {profile_fitted.name}\nAdded profile: {profile_added.name}")
        plt.legend(loc="upper left", fontsize=10)
        # Setting axis labels
        plt.xlabel(f"{column} threshold value")
        plt.ylabel("Completeness, purity and false detection rate (%)")
        # Adding the grid
        plt.grid(alpha=0.3)
        # Saving the figure
        plt.savefig(f"{output_folder}{COLUMN_FILE_LABELS[column]}.png", dpi=400)
        # Closing the figure
        plt.close()

        # Inform user
        tqdm.write(f"[INFO] Sample completeness for column {column} plot saved as `{output_folder}{COLUMN_FILE_LABELS[column]}.png`.")

    # Returning to the main function
    return

# Function to perform the whole mock analysis
def mock_analysis(profile_to_add : Profile, profile_to_fit : Profile) -> None:
    """
    This function performs the whole analysis with the mock spectra. It creates the mock spectra or
    load them from a pickle file if they already exist. It then performs the cross-correlation analysis 
    on them, if needed. It loads the results of this analysis and plots statistics and some spectre from it.

    :param profile_to_add: The profile to add to the mock spectra.
    :type profile_to_add: Profile
    :param profile_to_fit: The profile to fit to the mock spectra.
    :type profile_to_fit: Profile
    """
    
    # ====================
    # Base configuration
    # ====================

    # Inform user
    print("\n[INFO] Starting mock spectra analysis...")

    # Defining the output folder
    MOCK_ANALYSIS_FOLDER = f"{RESULTS_FOLDER}results/mock_analysis_ntot-{log10(profile_to_add.Ntot):.1f}vs{log10(profile_to_fit.Ntot):.1f}_b-{profile_to_add.b_param:.0f}vs{profile_to_fit.b_param:.0f}_Jmax-{profile_to_add.Jmax:.0f}vs{profile_to_fit.Jmax:.0f}/"

    # Creating outputfolder if it doesn't exist
    os.makedirs(MOCK_ANALYSIS_FOLDER, exist_ok=True)

    # Output file path
    output_file = os.path.join(MOCK_ANALYSIS_FOLDER, "mock_spectra.pk1")

    # ====================
    # Mock spectra creation and cross-correlation analysis
    # ====================

    # If the output folder is empty
    if len(os.listdir(MOCK_ANALYSIS_FOLDER)) == 0:
        # Calling the function to add the H₂ synthetic profile to the spectra
        mock_spectra, sample_spectra = create_mock_spectra(profile_to_add, output_file)
        # Calling the function to perform the cross-correlation analysis on the mock spectra sample
        run_cross_correlation_analysis(spectra_files=sample_spectra, profiles_to_fit=[profile_to_fit], output_folder=MOCK_ANALYSIS_FOLDER)
    
    # Else, loading the dictionary of the mock spectra from a local file
    else:
        # Reading the dictionary of the mock spectra from the pickle file
        with open(output_file, "rb") as file:
            mock_spectra = pickle.load(file)
        # Inform user
        print(f"[INFO] {len(mock_spectra)} mock spectra loaded successfully!")

    # ====================
    # Results analysis
    # ====================
    
    # Loading the results of the cross-correlation analysis from the output folder
    AnalysisResults.reload(folder=MOCK_ANALYSIS_FOLDER)

    # Calling the function to plot the statistics of the mock spectra used for the analysis
    mock_spectra_statistics(mock_spectra=mock_spectra, profile_to_fit=profile_to_fit, output_folder=MOCK_ANALYSIS_FOLDER)

    # Calling the function to perform the sample completeness and purity analysis
    sample_completeness_analysis(mock_spectra=mock_spectra, columns=[ColNames.CORR_PARAM, ColNames.CORR_COEFF, ColNames.CORE_TRANS, ColNames.GRADE], output_folder=MOCK_ANALYSIS_FOLDER)
