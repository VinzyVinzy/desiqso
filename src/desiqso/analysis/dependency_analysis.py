"""
This module contains functions to perform the dependency analysis of the synthetic profiles. In particular,
it computes the dependency of the correlation coefficient, correlation probability and core transmission
with SNR for a given synthetic profile. The median of the core transmission is also saved in a `.txt` file.
as it is the expected core transmission of the synthetic profile, and may need to be accounted for in 
the later analysis.
"""

# Packages import
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Local imports
from src.desiqso.analysis.absorption_masks import compute_h2_absorption_masks
from src.desiqso.analysis.cross_correlation import cross_correlate
from src.desiqso.config import (EXPECTED_CORE_TRANSMISSIONS_PATH, DEPENDECY_RESULTS_FOLDER, SPECTRA_DATA_FOLDER, settings)
from src.desiqso.models.profile import ProfileManager, Profile
from src.desiqso.models.spectrum import SpectrumRecord

# Function to compute the dependency of the correlation coefficient, correlation probability and core transmission
def compute_profile_snr_dependency(profile : Profile, show: bool = False, save: bool = True) -> None:
    """
    Compute the dependency of the correlation coefficient, correlation probability and core transmission 
    with SNR for a given synthetic profile.

    :param profile: The synthetic profile to compute the dependency for.
    :type profile: Profile
    :param show: Whether to show the plots or not.
    :type show: bool, optional
    :param save: Whether to save the plots or not.
    :type save: bool, optional
    :return: This function does not return anything, but save the plots (if required).
    :rtype: None
    """

    # ===================== 
    # Configuration
    # =====================

    # Update and apply matplotlib settings
    settings["xtick.top"] = True
    plt.rcParams.update(**settings)

    # Inform user
    print(f"\n[INFO] Computing the dependency of the correlation and core transmission with SNR for synthetic profile {profile.name}...")

    # Choose a random spectra from the folder to retrieve the wavelength array from
    random_file = np.random.choice(os.listdir(SPECTRA_DATA_FOLDER))
    # Open the FITS file as a `SpectrumRecord` instance to retrieve the relevant data
    with fits.open(os.path.join(f"{SPECTRA_DATA_FOLDER}/{random_file}"), memmap=False) as hdul:
        record = SpectrumRecord.from_fits(hdul)
        wavelength = record.wavelength
        base_redshift = record.redshift
        base_mask = record.mask

    # Rebin the synthetic profile to match the wavelength grid of the observed spectrum
    _, _, flux_rebinned = compute_h2_absorption_masks(wavelength, base_redshift, profile, base_mask)
    # Replace 0 values with 1 in the rebinned flux array to compute SNR correctly
    flux_rebinned[flux_rebinned == 0] = 1

    # Define the SNR range in which to compute the dependency
    snr_values = np.linspace(1.6, 20, 150)
    # Initialize output arrays
    correlation_values = np.full_like(snr_values, np.nan)
    correlation_probs  = np.full_like(snr_values, np.nan)
    core_transmissions = np.full_like(snr_values, np.nan)
    snr_true_values    = np.full_like(snr_values, np.nan)

    # ===================== 
    # Cross-correlation analysis
    # =====================

    # Get the index of the synthetic profile to retrieve the corresponding results
    profile_idx = ProfileManager.get_profile_index(profile.name)

    # Loop on SNR values
    for i, snr in enumerate(tqdm(snr_values, desc="Processing SNR values...", unit="SNR values")):
        # Compute the noise using the SNR target value and a Gaussian distribution
        sigma = 1/snr
        noise = np.random.normal(0, sigma, len(wavelength))
        # Add the noise to the rebinned flux to emulate a noisy spectrum containing H2
        new_flux_rebinned = flux_rebinned + noise
    
        # Create a `SpectrumRecord` instance to realise the cross-correlation analysis
        new_record = SpectrumRecord(wavelength=wavelength, flux=new_flux_rebinned, err=np.full_like(wavelength, sigma), redshift=base_redshift, redshift_err=0., redshift_wrn=0., name="Simulated", data_release="Simulated", model=np.ones_like(wavelength), mask=base_mask, spectype="Simulated", specid=0., ra=0., dec=0., chi_2=0., tsnr2_qso=0.)

        # Perform the cross-correlation analysis on the simulated spectrum
        result = cross_correlate(new_record)
        # Retrieve the relevant values from the cross-correlation analysis
        correlation_values[i] = result.best_correlation_values[profile_idx]
        correlation_probs[i]  = result.best_correlation_probs[profile_idx]
        core_transmissions[i] = result.best_core_transmissions[profile_idx]
        snr_true_values[i]    = result.snr
        
    # Ordering the arrays by SNR
    order = np.argsort(snr_true_values)
    correlation_values = correlation_values[order]
    correlation_probs  = correlation_probs[order]
    core_transmissions = core_transmissions[order]
    snr_true_values    = snr_true_values[order]

    # ===================== 
    # Plotting the results
    # =====================

    # Inform user
    print(f"\n[INFO] Plotting the results of the SNR dependency analysis for synthetic profile {profile_idx}...")

    # Creating the first plot of the evolution of the correlation and core transmission with SNR
    _, ax = plt.subplots(figsize=(9, 6))
    # Plotting the evolution of the correlation and core transmission with SNR
    ax.plot(snr_true_values, correlation_values, color='tab:blue', label="Correlation value")
    ax.plot(snr_true_values, core_transmissions, color='tab:orange', label="Core transmission")
    # Setting the axis labels
    ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    
    # Creating plot title
    title = "Evolution of Correlation and Core Transmission with SNR" + f"\n({profile.legend_label})"
    # Setting the plot title
    ax.set_title(title, fontsize=14)

    # Adding a grid
    ax.grid(True, alpha=0.3)
    # Adding a legend
    ax.legend()
    # Adjusting the layout
    plt.tight_layout()
    # Saving the plot, if requested
    if save:
        # Creating the folder if it doesn't exist
        output_path = os.path.join(DEPENDECY_RESULTS_FOLDER, f"snr-dependency/{profile.name}_correlation-coefficient_core-transmission.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Saving the plot
        plt.savefig(output_path)
    # Displaying the plot, if requested
    if show:
        plt.show()

    # Creating the second plot of the evolution of the correlation probability with SNR
    _, ax = plt.subplots(figsize=(9, 6))
    # Plotting the evolution of the correlation probability with SNR
    ax.plot(snr_true_values, np.log10(correlation_probs), color='tab:red', lw=2, label="log10(Correlation probability)")
    # Setting the axis labels
    ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}(P)$", fontsize=12)

    # Creating plot title
    title = "Evolution of Correlation Probability with SNR" + f"\n({profile.legend_label})"
    # Setting the plot title
    ax.set_title(title, fontsize=14)

    # Adding a grid
    ax.grid(True, alpha=0.3)
    # Adding a legend
    ax.legend()
    # Adjusting the layout
    plt.tight_layout()
    # Saving the plot, if requested
    if save:
        # Creating the folder if it doesn't exist
        output_path = os.path.join(DEPENDECY_RESULTS_FOLDER, f"snr-dependency/{profile.name}_correlation_probability.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Saving the plot
        plt.savefig(output_path)
    # Displaying the plot, if requested
    if show:
        plt.show()
    
    # Closing the plots to avoid memory issues
    plt.close('all')
    
    # Inform user
    print(f"[INFO] Dependency of the correlation and core transmission with SNR for synthetic profile {profile_idx} computed and plotted successfully!")
    
    # Saving the expected core transmission for each synthetic profile
    save_profile_expected_core_transmission(profile_name=profile.name, expected_core_transmission=np.median(core_transmissions))

# Function to save the expected core transmission of a synthetic profile, computed during the SNR dependency analysis
def save_profile_expected_core_transmission(profile_name : str, expected_core_transmission : float) -> None :
    """
    Save the expected core transmission of a synthetic profile in a `.txt` file. The file is in the format 
    "profile_name expected_core_transmission" and is saved in the `EXPECTED_CORE_TRANSMISSIONS_PATH` 
    directory where the output file is.

    If the file already exists, the function will replace the existing expected core transmission 
    associated with the synthetic profile name with the new one computed.

    If the file does not exist, the function will create a new one with the synthetic profile name and 
    its associated expected core transmission.

    :param profile_name: The name of the synthetic profile.
    :type profile_name: `str`
    :param expected_core_transmission: The expected core transmission of the synthetic profile.
    :type expected_core_transmission: `float`
    :return: This function does not return anything, but it saves the expected core transmission in a file.
    :rtype: None
    """
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(EXPECTED_CORE_TRANSMISSIONS_PATH), exist_ok=True)

    # Create a list to store the lines and a flag to check if the profile name was found
    lines = []
    found = False

    # Check if the output folder exists
    if os.path.exists(os.path.dirname(EXPECTED_CORE_TRANSMISSIONS_PATH)):
        # Read the existing file
        with open(EXPECTED_CORE_TRANSMISSIONS_PATH, "r", encoding="utf-8") as file:
            # Loop on the lines
            for line in file:
                # Split the line to retrieve the profile name associated to the expected core transmission
                name, *rest = line.strip().split("\t")
                # If the profile name is the same as the one of the synthetic profile
                if name == profile_name:
                    # Replace the expected core transmission
                    lines.append(f"{profile_name}\t{expected_core_transmission}\n")
                    found = True
                # Else, keep the existing line
                else:
                    lines.append(line)
    
    # If the profile name was not found in the existing file
    if not found:
        # Add the new line
        lines.append(f"{profile_name}\t{expected_core_transmission}\n")

    # Write the new file
    with open(EXPECTED_CORE_TRANSMISSIONS_PATH, "w", encoding="utf-8") as file:
        file.writelines(lines)
    
    # Inform the user
    print(f"\n[INFO] Expected core transmission for {profile_name} is {expected_core_transmission:.2f} and was saved in {EXPECTED_CORE_TRANSMISSIONS_PATH}\n")

    # Return to the main program
    return
