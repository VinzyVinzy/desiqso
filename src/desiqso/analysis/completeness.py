"""
This module contains functions to perform the completeness and purity analysis of the mock spectra sample.
"""

# Packages import
import matplotlib.pyplot as plt
from math import log10
import numpy as np
import os
import pandas as pd
import pickle
import random
from tqdm import tqdm

# Local imports
from src.desiqso.analysis.absorption_masks import compute_h2_absorption_masks
from src.desiqso.analysis.cross_correlation import run_cross_correlation_analysis
from src.desiqso.analysis.mocks_spectra import (sample_completeness_analysis, mock_spectra_statistics,)
from src.desiqso.config import (RESULTS_FOLDER, settings, SNR_THRESHOLD, REDSHIFT_RANGE,)
from src.desiqso.constants import (DESI_RESOLUTION_POWER, Categories, ColNames, Modes,)
from src.desiqso.data.loader import load_spectrum_from_filename
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.models.profile import  (ProfileManager, Profile,)
from src.desiqso.models.results import CrossCorrelationResult
from src.desiqso.models.spectrum import SpectrumRecord
from src.desiqso.utils.helpers import compute_column_weights

# Function to create a sample of mock spectra with the same SNR and redshift distribution as the real sample
def create_mock_spectra_sample(profile_to_add : Profile, folder : str) -> tuple[np.ndarray[SpectrumRecord], dict[str, SpectrumRecord]]:
    """
    This function creates a sample of mock spectra using spectra from the DESI 
    survey with high-SNR and low-chance of H₂. It adds a synthetic H₂ profile to
    some of them, selected randomly. It also adds noise to all of the spectra to
    simulate the SNR distribution of the real sample.

    :param profile_to_add: The synthetic H₂ profile to add to the mock spectra.
    :type profile_to_add: Profile
    :param folder: The folder to save the mock spectra.
    :type folder: str
    :return tuple[np.ndarray[SpectrumRecord], dict[str, SpectrumRecord]]: A tuple 
    containing the array of mock spectra and the dictionnary of the mock spectra
    in which H₂ was added.
    """

    # ====================
    # Sample selection
    # ====================

    # Inform user
    print("[INFO] Selecting sample for mock spectra creation...")

    # Loading the cross-correlation analysis results
    AnalysisResults.reload(verbose=False)
    # Retrieving the table corresponding to the spectra with high SNR and with low chances of having H₂
    table = AnalysisResults.results_survey(mode=Modes.ALL, profile_name="best", thresholds_dict={ColNames.SNR : (15, None), ColNames.QSO_Z : (REDSHIFT_RANGE[0], None)})
    # Selecting only spectra that are not from the "confirmed", "borderline" or "rejected" categories
    table = table[table[ColNames.CATEGORY].isin([Categories.OTHER, Categories.REJECTED])]
    # List of the filenames of the spectra selected
    filenames = table[ColNames.FILENAME].tolist()
    # List of 1000 randomly selected filenames to add the synthetic profile
    mock_spectra_list = np.random.choice(filenames, size=1000, replace=False).tolist()
    # Create complete synthetic profile to add to the mock spectra
    profile_to_add = profile_to_add.get_complete_profile()

    # ====================
    # Creating and saving the mock spectra
    # ====================

    # Initialize dictionary containing the filenames and corresponding records of the mock spectra
    mock_spectra = []
    # Initialize dictionary containing the filenames and corresponding records of the real spectra
    true_mock_spectra = {}

    # Looping on the filenames
    for filename in tqdm(filenames, desc="Creating mock spectra...", unit="spectra"):

        # Opening the `.fits` file corresponding as a `SpectrumRecord` instance
        record = load_spectrum_from_filename(filename)

        # If the spectrum is selected to become a true mock spectrum
        if filename in mock_spectra_list:
            # Rebinning the synthetic profile to match the record
            mask_data, _, flux_rebinned = compute_h2_absorption_masks(record.wavelength, record.redshift, profile_to_add, record.mask)
            # Find the indices of the data points that are not masked
            indices = np.where(mask_data)[0]
            # Creating the slice for the synthetic profile
            indices_slice = slice(max(0, indices[0]-100), indices[-1]+100)
            # Normalizing the spectrum
            continuum = record.continuum
            record.flux /= continuum
            # Adding the synthetic profile to the original spectrum
            record.flux[indices_slice] *= flux_rebinned[indices_slice]
            # Returning to the original units
            record.flux *= continuum
            
        # Selecting a random SNR
        snr = random.choice(AnalysisResults._results[ColNames.SNR])
        # If the target SNR is greater than the current SNR, continue
        if snr > record.snr:
            mock_spectra.append(record)
            continue
        # Computing the noise
        sigma = np.abs(record.flux)/snr
        noise = np.random.normal(0, sigma, len(record.wavelength))
        # Adding the noise to the flux
        record.flux += noise
        # Updating the error
        record.err = np.full_like(record.err, sigma)
        # If the SNR is below the threshold
        if record.snr < SNR_THRESHOLD:
            continue
        # Adding the record to the list
        mock_spectra.append(record)
        # If the spectrum is selected to become a true mock spectrum
        if filename in mock_spectra_list:
            # Adding it to the dictionary
            true_mock_spectra[filename] = record

    # Converting the list to an array for easier handling
    mock_spectra = np.asarray(mock_spectra)
    
    # ====================
    # Plotting the SNR and redshift distributions to ensure they are similar
    # ====================

    # Turn interactive mode off
    plt.ioff()

    # Updating the matplotlib settings
    settings["xtick.top"] = True
    plt.rcParams.update(**settings)
    # Creating the figure and axis
    _, ax = plt.subplots(figsize=(12,8))
    # Setting the labels and title of the plot
    plt.xlabel(ColNames.SNR)
    plt.ylabel("Percentage of total spectra")
    plt.title("Distribution of SNR for both samples")
    # Setting the limits of the plot
    plt.xlim(0, 20)
    # Converting the records properties to a DataFrame
    sample_data = pd.DataFrame({ColNames.SNR : [record.snr for record in mock_spectra], ColNames.Z : [record.redshift for record in mock_spectra]})
    # Selecting the part of the table with non-NaN values for the column to plot
    sample_data_to_plot = sample_data.dropna(subset=[ColNames.SNR, ColNames.Z])
    sample_data_to_plot[ColNames.SNR].hist(bins=np.arange(0, 20.25, 0.25), weights=compute_column_weights(sample_data_to_plot, ColNames.SNR), ax=ax, edgecolor='black', label="Mock spectra sample", histtype='step', linewidth=2)
    # Displaying the number of spectra used
    ax.text(0.90, 0.95,f"N = {len(sample_data_to_plot)}", transform=ax.transAxes, fontsize=12, va='top')
    # Selecting the part of the table with non-NaN values for the column to plot
    data_to_plot = AnalysisResults._results.copy().dropna(subset=[ColNames.SNR, ColNames.Z])
    data_to_plot[ColNames.SNR].hist(bins=np.arange(0, 20.25, 0.25), weights=compute_column_weights(data_to_plot, ColNames.SNR), ax=ax, edgecolor='red', label="DESI sample", histtype='step', linewidth=2)
    # Displaying the number of spectra used
    ax.text(0.90, 0.90,f"N = {len(data_to_plot)}", transform=ax.transAxes, fontsize=12, va='top')
    # Setting the legend
    ax.legend(loc='upper right')
    # Saving the figure
    plt.savefig(f"{folder}SNR_distribution.png", dpi=300)
    # Closing figure
    plt.close()

    # Creating the figure and axis
    _, ax = plt.subplots(figsize=(12,8))
    # Setting the labels and title of the plot
    plt.xlabel(ColNames.Z)
    plt.ylabel("Percentage of total spectra")
    plt.title("Distribution of redshifts for both samples")
    # Setting the limits of the plot
    plt.xlim(2.5, 6)
    # Selecting the part of the table with non-NaN values for the column to plot
    sample_data_to_plot[ColNames.Z].hist(bins=np.arange(2.5, 6, 0.10), weights=compute_column_weights(sample_data_to_plot, ColNames.Z), ax=ax, edgecolor='black', label="Mock spectra sample", histtype='step', linewidth=2)
    # Displaying the number of spectra used
    ax.text(0.90, 0.95,f"N = {len(sample_data_to_plot)}", transform=ax.transAxes, fontsize=12, va='top')
    # Selecting the part of the table with non-NaN values for the column to plot
    data_to_plot[ColNames.Z].hist(bins=np.arange(2.5, 6, 0.1), weights=compute_column_weights(data_to_plot, ColNames.Z), ax=ax, edgecolor='red', label="DESI sample", histtype='step', linewidth=2)
    # Displaying the number of spectra used
    ax.text(0.90, 0.90,f"N = {len(data_to_plot)}", transform=ax.transAxes, fontsize=12, va='top')
    # Setting the legend
    ax.legend(loc='upper right')
    # Saving the figure
    plt.savefig(f"{folder}redshift_distribution.png", dpi=300)
    # Closing figure
    plt.close()

    # Saving the dictionary of the mock spectra in a pickle file
    with open(f"{folder}true_mock_spectra.pkl", "wb") as file:
        pickle.dump(true_mock_spectra, file)
    # Saving the array of the mock spectra in a pickle file
    with open(f"{folder}mock_spectra.pkl", "wb") as file:
        pickle.dump(mock_spectra, file)

    # Returning the array containing the mock spectra and the dictionary containing the true mock spectra
    return mock_spectra, true_mock_spectra

# Function to perform the complete mock analysis for a given total column density
def completeness_analysis(mock_spectra : np.ndarray[SpectrumRecord], true_mock_spectra : dict[str, SpectrumRecord], folder : str, profile_added : Profile, profile_to_fit : Profile) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    """
    This function performs the cross-correlation analysis of the mock spectra 
    sample, plots the statistics of the mock spectra used for the analysis, 
    performs the sample completeness and purity analysis and finally computes
    the number of true positives, false positives, true negatives and false negatives
    in the mock spectra sample for different SNR thresholds.

    :param mock_spectra: Array containing the mock spectra sample
    :type mock_spectra: np.ndarray[SpectrumRecord]
    :param true_mock_spectra: Dictionnary containing the mock spectra
    in which H₂ was added
    :type true_mock_spectra: dict[str, SpectrumRecord]
    :param folder: Output folder
    :type folder: str
    :param profile_added: Profile added to the mock spectra
    :type profile_added: Profile
    :param profile_to_fit: Profile to fit
    :type profile_to_fit: Profile
    :return tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]: 
    Tuple containing the number of true positives, false positives, true negatives and false negatives in the mock spectra sample for different SNR thresholds
    """

    # If the cross-correlation analysis was not performed yet
    if len(os.listdir(folder)) < 5:
        # Reset the results files of the cross-correlation analysis
        CrossCorrelationResult._results = None
        # Perform the cross-correlation analysis on the mock spectra sample
        run_cross_correlation_analysis(spectra_files=mock_spectra, profiles_to_fit=[profile_to_fit], output_folder=folder)

    # Load the results of the cross-correlation analysis
    AnalysisResults.reload(folder, verbose=False)

    # Calling the function to plot the statistics of the mock spectra used for the analysis
    mock_spectra_statistics(mock_spectra=true_mock_spectra, profile_to_fit=profile_to_fit, output_folder=f"{folder}threshold_analysis/")
    
    # Calling the function to perform the sample completeness and purity analysis
    sample_completeness_analysis(true_mock_spectra, f"{folder}threshold_analysis/", profile_added, profile_to_fit)

    # Compute the completeness and purity of the analysis for different SNR
    true_positives, false_positives, true_negatives, false_negatives = snr_sample_completeness(true_mock_spectra)

    # Returning the dictionnaries containing the true positives, false positives, true negatives and false negatives for different SNR thresholds
    return true_positives, false_positives, true_negatives, false_negatives

# This function computes the number of true positives, false positives, true negatives and false negatives in the mock spectra sample for different SNR thresholds
def snr_sample_completeness(mock_spectra : dict[str, SpectrumRecord]) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    """
    This function computes the number of true positives, false positives, true negatives 
    and false negatives in the mock spectra sample for different SNR thresholds. These 
    values are later used for computing the completeness and purity of the analysis.

    :param mock_spectra: Dictionnary containing the mock spectra
    in which H₂ was added
    :type mock_spectra: dict[str, SpectrumRecord]
    :return tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]: 
    Tuple containing the number of true positives, false positives, true negatives and false negatives in the mock spectra sample for different SNR thresholds
    """

    # Retrieve the mock spectra analysis results
    results = AnalysisResults._results.copy()
    # Mask to select only the spectra in which there is H₂ to find
    mask_true = (results[ColNames.FILENAME].isin(mock_spectra.keys()))
    # Mask to select only the spectra in which H₂ was detected by the algorithm
    mask_valid = (results[ColNames.IS_VALID] == 1)

    # Initializing the dictionnaries containing the completeness and purity for different SNR
    true_positives = {}
    true_negatives = {}
    false_positives = {}
    false_negatives = {}

    # Looping over some SNR values
    for snr in [SNR_THRESHOLD, 2, 3, 5, 7] :
        # Mask to select only the spectra with a SNR greater than the current SNR value
        mask_snr = (results[ColNames.SNR] >= snr)
        # Computing the number of true positive
        true_positives[snr] = np.sum(mask_true & mask_snr & mask_valid)
        # Computing the number of false positive
        false_positives[snr] = np.sum(~mask_true & mask_snr & mask_valid)
        # Computing the number of true negative
        true_negatives[snr] = np.sum(mask_true & mask_snr & ~mask_valid)
        # Computing the number of false negative
        false_negatives[snr] = np.sum(~mask_true & mask_snr & ~mask_valid)

    # Returning the dictionnaries containing the number of true positives, false positives, true negatives and false negatives for different SNR thresholds
    return true_positives, false_positives, true_negatives, false_negatives

# Function to plot the completeness and purity of the mock spectra sample
def plot_completeness_purity(all_true_positives : list[dict[int, float]], all_false_positives : list[dict[int, float]], all_true_negatives : list[dict[int, float]], all_false_negatives : list[dict[int, float]], total_column_densities : np.ndarray[float], profile_to_fit : Profile, folder : str):
    """
    This function plots the evolution of the completeness and purity of the mock 
    spectra sample cross-correlation analysis as a function of the total column 
    density and the SNR threshold. Also saves the values to a file for manual 
    inspection.

    :param all_true_positives: List containing the dictionnaries containing the number of true positives in the mock spectra sample for different SNR thresholds.
    :type all_true_positives: list[dict[int, float]]
    :param all_false_positives: List containing the dictionnaries containing the number of false positives in the mock spectra sample for different SNR thresholds.
    :type all_false_positives: list[dict[int, float]]
    :param all_true_negatives: List containing the dictionnaries containing the number of true negatives in the mock spectra sample for different SNR thresholds.
    :type all_true_negatives: list[dict[int, float]]
    :param all_false_negatives: List containing the dictionnaries containing the number of false negatives in the mock spectra sample for different SNR thresholds.
    :type all_false_negatives: list[dict[int, float]]
    :param total_column_densities: Array containing the total column density values.
    :type total_column_densities: np.ndarray[float]
    :param profile_to_fit: Profile used for the cross-correlation analysis.
    :type profile_to_fit: Profile
    :param folder: Output folder.
    :type folder: str
    """

    # Turn interactive mode off
    plt.ioff()

    # Updating the matplotlib settings
    settings["xtick.top"] = True
    plt.rcParams.update(**settings)

    # Dictionnary for the colors
    colors = {SNR_THRESHOLD : "black", 2 : "blue", 3 : "green", 5 : "orange", 7 : "red"}

    # Computing the completenesses and purities from the values
    completenesses = [{snr : (all_true_positives[i][snr] / (all_true_positives[i][snr] + all_true_negatives[i][snr])) for snr in all_true_positives[i].keys()} for i in range(len(all_true_positives))]
    purities = [{snr : (1 - (all_false_positives[i][snr] / (all_true_positives[i][snr] + all_false_positives[i][snr]))) for snr in all_true_positives[i].keys()} for i in range(len(all_true_positives))]
    false_detection_rates = [{snr : (all_false_positives[i][snr] / (all_false_positives[i][snr] + all_false_negatives[i][snr])) for snr in all_true_positives[i].keys()} for i in range(len(all_true_positives))]

    # Loop over completenesses and purities
    for key, values_list in {"Completeness" : completenesses, "Purity" : purities, "False detection rate" : false_detection_rates}.items():

        # Creating the figure and axis
        _, ax = plt.subplots(figsize=(12,8))

        # Setting the labels and title of the plot
        plt.xlabel("Total column density")
        plt.ylabel(f"{key} (%)")
        plt.title(rf"{key} for the base program and with thresholds SNR $\geq$ {list(values_list[0].keys())[1:]}"+f"\nProfile: {profile_to_fit.name}")
        # Setting the limits of the plot
        plt.xlim(np.min(np.log10(total_column_densities))-0.1, np.max(np.log10(total_column_densities))+0.1)
        plt.ylim(-5, 105)

        # Looping over SNR values
        for snr in values_list[0].keys():
            # Retrieving the values for the current SNR
            values = [values_dict[snr]*100 for values_dict in values_list]
            # Plotting the values
            ax.plot(np.log10(total_column_densities), values, label=(f"SNR >= {snr}" if snr != SNR_THRESHOLD else "Base algorithm"), color=colors[snr], alpha=0.7)

        # Adding the legend
        plt.legend()
        plt.grid(alpha=0.3)

        # Saving the plot
        plt.savefig(f"{folder}{"-".join(key.lower().split(" "))}_vs_total-col-density.png")
        plt.close()
    
    # Saving the results of the analysis in the dedicated file
    with open(f"{folder}completeness_purity_analysis.txt", "w") as file:
        # Writing the header
        file.write("\t".join(["log(N)", "SNR", "True positives", "True negatives", "False positives", "False negatives", "Completeness (%)", "Purity (%)", "False detection rate (%)"]) + "\n")
        # Looping over the total column densities
        for i in range(len(total_column_densities)):
            # Looping over the SNR values
            for snr in completenesses[0].keys():
                file.write(f"{log10(total_column_densities[i])}\t{snr}\t{all_true_positives[i][snr]}\t{all_true_negatives[i][snr]}\t{all_false_positives[i][snr]}\t{all_false_negatives[i][snr]}\t{completenesses[i][snr]*100}\t{purities[i][snr]*100}\t{false_detection_rates[i][snr]*100}\n")

    # Returning to the main function
    return

# Function to run the program completeness and purity analysis
def run_completeness_analysis(profile_to_fit : Profile) -> None:
    """
    This function runs the complete mock sample analysis to inspect the performance of the algorithm.
    It loops over different values of the total column density, creates a mock sample reproducing the
    SNR distribution of the DESI sample and injects some H₂ into some of the spectra. It then performs
    the cross-correlation analysis over the mock spectra sample and computes the completeness and purity 
    of the algorithm.

    :param profile_to_fit: Profile to fit
    :type profile_to_fit: Profile
    :return None: The function does not return anything.
    """

    # ====================
    # Base configuration
    # ====================

    # Inform user
    print("\n[INFO] Starting program completeness analysis using mock spectra...")

    # Defining the output folder
    ANALYSIS_FOLDER = f"{RESULTS_FOLDER}results/completeness-analysis_{log10(profile_to_fit.Ntot):.1f}/"
    os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

    # Loading all synthetic H₂ profiles
    ProfileManager.load_all(verbose=False)

    # ====================
    # Completeness analysis
    # ====================

    # List of total column density values to perform the completeness analysis on
    total_column_densities = 10**np.linspace(18, 20.5, 6)

    # Initializing lists to store the completeness and purity values
    all_true_positives = []
    all_false_positives = []
    all_true_negatives = []
    all_false_negatives = []

    # Looping over the total column densities
    for density in total_column_densities:

        # Inform user
        print(f"\n[INFO] Analyzing mock spectra with total column density {density:.2e} cm^-2...")

        # Defining the output subfolder
        subfolder = f"{ANALYSIS_FOLDER}{log10(density):.1f}/"
        os.makedirs(subfolder, exist_ok=True)

        # Creating a synthetic profile with the given total column density
        profile_to_add = Profile.from_synthetic(
            resolution_power= DESI_RESOLUTION_POWER,# Resolution power of the synthetic profile
            pixel_size      = 5.,                   # Pixel size in km/s for the synthetic profile
            T_exc0          = 75.,                  # Excitation temperature of the J=0 level in K
            Jmax            = 1,                    # Maximum rotational level to include in the synthetic profile (0 and 1 by default)
            Ntot            = density,              # Total column density of H₂ in cm^-2
            b_param         = 3.,                   # Doppler parameter for the synthetic profile in km/s
            save            = False,                # Do not save the profile
            verbose         = False                 # Do not print information about the profile
        )

        # If the output folder is empty
        if len(os.listdir(subfolder)) == 0:
            # Calling the function to create a sample of mock spectra with the same SNR and redshift distribution as the real sample
            mock_spectra, true_mock_spectra = create_mock_spectra_sample(profile_to_add, subfolder) 
        # Else, load directly the mock spectra sample and the dictionnary containing the real mock spetcra from local files
        else :
            # Saving the dictionary of the mock spectra in a pickle file
            with open(f"{subfolder}true_mock_spectra.pkl", "rb") as file:
                true_mock_spectra =pickle.load(file)
            # Saving the array of the mock spectra in a pickle file
            with open(f"{subfolder}mock_spectra.pkl", "rb") as file:
                mock_spectra = pickle.load(file)
            # Inform user
            print(f"\n[INFO] Loaded {len(mock_spectra)} mock spectra sample from {subfolder}, with {len(true_mock_spectra)} real mock spectra.")

        # Calling the function to perform the completeness analysis on the sample of mock spectra
        true_positives, false_positives, true_negatives, false_negatives = completeness_analysis(mock_spectra, true_mock_spectra, subfolder, profile_to_add, profile_to_fit)

        # Appending the samples of true positives, false positives, true negatives and false negatives to their respective lists
        all_true_positives.append(true_positives)
        all_false_positives.append(false_positives)
        all_true_negatives.append(true_negatives)
        all_false_negatives.append(false_negatives)
    
    # Calling the function to plot the completeness and purity values for different total column densities and SNR
    plot_completeness_purity(all_true_positives, all_false_positives, all_true_negatives, all_false_negatives, total_column_densities, profile_to_fit, ANALYSIS_FOLDER)
