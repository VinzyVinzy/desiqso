"""
This module contains functions to plot spectra and profiles after the cross-correlation analysis.
It allows the user to visualize the best-fit profiles and the corresponding spectra, in order to 
check the quality of the results.
"""

# Packages import
from astropy.convolution import convolve, Gaussian1DKernel
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# Local imports
from src.desiqso.analysis.absorption_masks import compute_h2_absorption_masks
from src.desiqso.config import (SPECTRA_PLOTS_FOLDER, MULTIPLY_BY_CONTINUUM, settings)
from src.desiqso.constants import (ColNames, Modes,)
from src.desiqso.data.loader import load_spectrum_from_filename
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.models.profile import ProfileManager
from src.desiqso.utils.helpers import compute_grade

# Function plotting a spectrum
def plot_spectrum(row : pd.Series, folderpath : str) -> None :
    """
    Plot the spectrum of a DESI quasar given a `SpectrumRecord` instance.

    This function will plot the raw and smoothed spectrum of the given quasar, and
    overplot the best-match synthetic H₂ profile if available. It also displays the 
    constant continuum level used for the cross-correlation analysis (if any was used) 
    and a scatter of the pixels used for the correlation and for the core transmission 
    computing. The x-axis range is set to rest-frame 1000-1300, and a secondary axis 
    is added to show the rest-frame wavelength associated with each observed wavelength 
    value. The resulting plot is shown with a legend.

    :param record: The `SpectrumRecord` instance containing the spectrum data to plot.
    :type record: SpectrumRecord
    :param best_redshift: The best redshift value for the cross-correlation analysis.
    :type best_redshift: float
    :param profile: The `Profile` instance representing the synthetic H₂ profile to use.
    :type profile: `Profile`
    :param show: Flag indicating whether to show the plot. Default is True.
    :type show: bool
    :param save: Flag indicating whether to save the plot. Default is True.
    :type save: bool
    """

    # ============
    # Configuration
    # ============

    # Obtaining the SpectrumRecord instance for the spectrum
    record = load_spectrum_from_filename(row[ColNames.FILENAME])
    
    # Obtaining the best redshift and profile for the spectrum
    profile = ProfileManager.get(row[ColNames.PROFILE])

    # Turn off interactive mode to prevent displaying bugs
    plt.ioff()

    # Update plot settings
    settings["ytick.right"] = True
    plt.rcParams.update(**settings)

    # ============
    # Results retrieval
    # ============

    # Retrieve best fit redshift and core transmissions of the spectrum
    best_redshift      = row[ColNames.Z]
    core_transmissions = row[ColNames.J_CORE_TRANS]
    correlation_param  = row[ColNames.CORR_PARAM]
    core_transmission  = row[ColNames.CORE_TRANS]
    # Compute the fit grade
    grade = compute_grade(core_transmissions)

    # ===========
    # Plotting
    # ===========

    # Initialize figure
    fig, ax = plt.subplots(figsize=(16,8))

    # Initialize the plot title
    title = rf"{record.name} ($z_{{QSO}}$={record.redshift:.3f})"
    title += f"\nGrade = {grade}, Correlation parameter={correlation_param:.3f}, Core transmission={core_transmission:.3f}, SNR={record.snr:.2f}"
    # Add the title to the plot
    plt.title(title, loc='left')
    # Setting the axis labels
    plt.xlabel(r"Observed wavelength $(\AA)$")
    plt.ylabel(r"$f_{\lambda}\ [10^{-17}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$")

    # Plot raw spectrum
    plt.plot(record.wavelength, record.flux, color="k", alpha=0.2, label="Unsmoothed spectrum")

    # Overplot spectrum smoothed (raw spectrum convolved with a 1D Gaussian Kernel using astropy)
    plt.plot(record.wavelength, convolve(record.flux, Gaussian1DKernel(3)), color="k", label="Smoothed Spectrum")

    # Overplot continuum model found in the SPARCL database
    plt.plot(record.wavelength, record.model, color="r", alpha=0.3, label="Continuum Model (DESI)")

    # Overplot the constant continuum level estimated, if used for the cross-correlation analysis
    if MULTIPLY_BY_CONTINUUM:
        plt.axhline(y=record.continuum, color="g", linestyle="--", alpha=0.3, label=f"Constant Continuum ({record.continuum:.2f})")

    # Set x-axis range to rest-frame 1000-1300 A
    plt.xlim(1000*(1+record.redshift), 1300*(1+record.redshift))

    # Set y-axis range using quantiles and an offset
    q_low, q_high = np.percentile(record.flux, [0.3, 99.7])
    plt.ylim(q_low-0.2, q_high+0.2)

    # Compute masks for H₂ absorption features and synthetic profile
    mask_data, mask_core, h2_synthetic_flux_rebinned = compute_h2_absorption_masks(record.wavelength, best_redshift, profile, record.mask)
    # If a constant continuum was applied for the cross-correlation analysis, multiply the synthetic profile by it
    continuum_level = record.continuum if MULTIPLY_BY_CONTINUUM else 1.
    # If spectrum processing was successful, overplot best-match synthetic profile
    plt.plot(record.wavelength, continuum_level*h2_synthetic_flux_rebinned, alpha=0.5, color="b", label=rf"Best fit $H_{{2}}$ profile (z~{best_redshift:.3f})")
    # Overplot absorption features used for the cross-correlation analysis
    plt.scatter(record.wavelength[mask_data], continuum_level*h2_synthetic_flux_rebinned[mask_data], alpha=0.3, color="b", s=10)
    # Overplot absorption features used for the cross-correlation analysis
    plt.scatter(record.wavelength[mask_core], continuum_level*h2_synthetic_flux_rebinned[mask_core], alpha=0.3, color="r", s=10)
    
    # Displaying the information
    label = profile.legend_label
    ax.text(0.02, 0.07, label, transform=ax.transAxes, fontsize=10, va='top')

    # Adding a top axis for rest-frame wavelength
    axis = plt.gca()
    # Functions to convert between rest-frame and observed wavelength
    def rest_to_obs(x):
        return x*(1+record.redshift)
    def obs_to_rest(x):
        return x/(1+record.redshift)
    # Secondary axis creation
    secax = axis.secondary_xaxis("top", functions=(obs_to_rest, rest_to_obs))
    secax.set_xlabel(r"Rest-frame wavelength $(\AA)$")
    
    # Adding legend and showing plot (if requested)
    plt.legend(loc="lower right", fontsize=10)

    # ==================
    # Saving plot
    # ==================
    
    # If the directory does not exist, create it
    os.makedirs(os.path.join(SPECTRA_PLOTS_FOLDER, f"{folderpath}/"), exist_ok=True)
    # Saving plot
    plt.savefig(os.path.join(SPECTRA_PLOTS_FOLDER, f"{folderpath}/") + f"{record.filename[:-4]}_{profile.name}.png", bbox_inches='tight', dpi=400)
    
    # Closing plot
    plt.close(fig)

# Function plotting all spectra
def plot_spectra(mode : str = Modes.ALL, thresholds_dict : dict = {}) -> None:
    """
    Function plotting multiple spectra. It loads the cross-correlation analysis results, loads the 
    spectra data from local files and plots them with the best-match synthetic profile for each 
    spectrum. Only a subset of the spectra can be plotted, depending on the selected mode.
    Available modes are: "all", "random", "preliminary", "confirmed", "borderline", "rejected",
    "valid" and "new" (see function `get_spectra_list(mode: str)` for more details).

    :param MODE: Mode of selection. Can be "all", "random", "preliminary", "confirmed", "borderline", "rejected", "valid" or "new".
    :type MODE: str
    """

    # Loading cross-correlation analysis results
    AnalysisResults.load_results()
    # Loading synthetic H₂ profiles
    ProfileManager.load_all()

    # Retrieving the table
    table = AnalysisResults.results_survey(mode = mode, profile_name="all", thresholds_dict=thresholds_dict,)

    # Define folderpath using the mode and thresholds
    folderpath = f"{mode}"
    # Adding thresholds
    for key, value in thresholds_dict.items():
        # Skipping None values
        if value == (None, None):
            continue
        # Updating folderpath
        folderpath += f"_{key}-{value[0]}-{value[1]}"

    # Loop on the list of valid spectra
    for _, row in tqdm(table.iterrows(), total=len(table), desc="Plotting valid spectra", unit="spectra"):

        # Plotting the spectrum using the dedicated function, without showing the plot
        plot_spectrum(row, folderpath)

    # Inform user
    print("[INFO] All valid spectra have been successfully plotted.\n")
    