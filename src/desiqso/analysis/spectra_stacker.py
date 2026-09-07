"""
This module contains functions to create and plot stacked spectra from a set of individual spectra. 

The main function, `spectra_stacker`, retrieves the analysis results, loads the synthetic H₂ profiles, 
and computes the stacked spectra using different stacking methods (raw, inverse variance weighted, mean,
 and median). The resulting stacked spectra are then plotted and saved in the specified output folder.
"""

# Packages import
from astropy.convolution import (convolve, Gaussian1DKernel,)
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from VoigtFit.container.lines import lineList

# Local imports
from src.desiqso.analysis.absorption_masks import compute_h2_absorption_masks
from src.desiqso.data.synthetic_profiles import wl_grid_const_speed
from src.desiqso.config import (settings, SPECTRA_PLOTS_FOLDER,)
from src.desiqso.constants import (ColNames, Modes, COLUMN_FILE_LABELS,)
from src.desiqso.data.loader import load_spectrum_from_filename
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.models.profile import ProfileManager
from src.desiqso.utils.helpers import compute_constant_continuum

# Do not show numpy warnings (neccessary for the VoigtFit package)
import warnings
warnings.filterwarnings("ignore",
    message="Input line 1 contained no data.*",
    category=UserWarning
)

# Function to create a stacked spectrum of the provided spectra
def plot_stacked_spectra(data : pd.DataFrame, output_folder : str, profile_name : str) -> None:
    """
    This function creates a stacked spectrum of the provided spectra using different 
    stacking methods (raw, inverse variance weighted, mean, and median). The stacked
    spectrum is then plotted in different wavelength windows for better visualisation 
    of the metal lines and comparison with the synthetic H₂ profile used for the 
    cross-correlation analysis.

    :param data: A pandas DataFrame containing the spectra data to be stacked.
    :type data: pd.DataFrame
    :param output_folder: The folder where the stacked spectrum plots will be saved.
    :type output_folder: str
    :param profile_name: The name of the synthetic profile used for the cross-correlation analysis.
    :type profile_name: str
    """

    # Inform user
    print("\n[INFO] Plotting stacked spectrum using different methods...")

    # Retrieving the synthetic profile corresponding to the provided profile name
    profile = ProfileManager.get(profile_name)

    # Creating a wavelength grid with a constant velocity step (i.e. constant pixel size in speed (5 km/s))
    rest_grid = wl_grid_const_speed(800., 2500., 5.)

    # Initializing the lists of the spectra fluxes and inverse variances
    fluxes = []
    ivars = []

    # Looping over the table to plot the stacked spectrum
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Retrieving spectra data for stacking", unit="spectra"):

        # Loading the spectrum
        spectrum = load_spectrum_from_filename(row[ColNames.FILENAME])

        # Converting the wavelength to rest frame of the absorber
        wave_rest = spectrum.wavelength / (1 + row[ColNames.Z])

        # Interpolating the spectrum flux and inverse variance to the common grid
        flux_interp = np.interp(rest_grid, wave_rest, spectrum.flux, left=np.nan, right=np.nan)
        ivar_interp = np.interp(rest_grid, wave_rest, spectrum.ivar, left=0, right=0)

        # Appending the flux  and inverse variance rebinned on the common grid to their respective list
        fluxes.append(flux_interp)
        ivars.append(ivar_interp)
        
    # Converting the fluxes, inverse variances and continua lists into numpy arrays for easier handling
    fluxes = np.asarray(fluxes)
    ivars  = np.asarray(ivars)
    
    # Creating useful masks for easier handling
    valid = np.isfinite(fluxes) & (ivars > 0)
    n_valid = np.sum(valid, axis=0)

    # Computing the raw stack flux (sum of all the fluxes)
    raw_flux = np.nansum(np.where(valid, fluxes, np.nan), axis=0)
    # Computing the raw stack variance
    vars = np.divide(1., np.where(valid, ivars, np.nan), out=np.zeros_like(ivars), where=valid)
    raw_var = np.sum(vars, axis=0)
    # Computing the raw stack error from the variance
    raw_err = np.sqrt(raw_var)

    # Computing the inverse variance summed stack
    ivar_sum = np.sum(vars, axis=0)
    # Computing the inverse variance weighted stacked flux
    ivar_weighted_flux = np.sum(fluxes * vars, axis=0) / ivar_sum
    # Computing the inverse variance weighted stacked error
    ivar_weighted_err = np.sqrt(np.divide(1.0, ivar_sum, out=np.full_like(ivar_sum, np.nan), where=ivar_sum > 0))

    # Computing the mean stack flux (mean of all the fluxes)
    mean_flux = np.nanmean(fluxes, axis=0)
    # Computing the mean stack variance
    mean_var = np.sum(vars, axis=0) / (n_valid**2)
    # Computing the mean stack error from the variance
    mean_err = np.sqrt(mean_var)

    # Computing the median stack flux (median of all the fluxes)
    med_flux = np.nanmedian(fluxes, axis=0)
    # Computing the median absolute deviation to use as a proxy for the variance
    mad = np.nanmedian(np.abs(fluxes - med_flux), axis=0)
    # Computing the median stack error from the variance
    med_err = 1.4826 * mad / np.sqrt(np.maximum(n_valid, 1))

    # Creating a list of tuple containing the name of the stack, the flux and the error associated for plotting
    stacks = [
        ("Raw", raw_flux, raw_err),
        ("Inverse variance weighted", ivar_weighted_flux, ivar_weighted_err),
        ("Mean", mean_flux, mean_err),
        ("Median", med_flux, med_err),
    ]

    # Turn off interactive mode to prevent visual artifacts
    plt.ioff()
    # Update plot settings
    settings["ytick.right"] = True
    settings["xtick.top"]   = True
    plt.rcParams.update(**settings)

    # Inform user
    print("\n[INFO] Plotting stacked spectrum...")

    # Looping over the stacks computed for plotting the resulting spectrum, with the corresponding metal lines
    for name, flux, err in tqdm(stacks, total=len(stacks), desc="Plotting stacked spectrum", unit="stacks"):

        # Computing the the continuum of the stacked spectrum using the associated function
        continuum = compute_constant_continuum(rest_grid, flux, err, 0.0)

        # Initialize figure
        _, ax = plt.subplots(figsize=(24,10))

        # Plotting the raw stacked spectrum
        plt.plot(rest_grid, flux, color="k", alpha=0.2, label="Unsmoothed stacked spectrum")        
        # Overplot the smoothed stacked spectrum (raw stacked spectrum convolved with a 1D Gaussian Kernel using astropy.convolution)
        plt.plot(rest_grid, convolve(flux, Gaussian1DKernel(3)), color="k", label="Smoothed stacked spectrum")

        # Setting the axis labels
        plt.xlabel(r"Rest-frame wavelength $(\AA)$")
        plt.ylabel(r"$f_{\lambda}\ [10^{-17}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}]$")
        
        # Displaying the information about the number of spectra stacked
        ax.text(0.02, 0.05, f"N = {len(data)} spectra", transform=ax.transAxes, fontsize=16, va='top')
        # Adding legend
        plt.legend(loc="lower right", fontsize=10)
        # Adding a grid
        plt.grid(True, alpha=0.1)

        # Defining the list of windows to plot for better visualisation of the lines
        windows = [
            ("Metal Window", 1240, 1390),
            ("Metal Window", 1390, 1540),
            ("Metal Window", 1540, 1690),
            ("Metal Window", 1690, 1840),
            ("Metal Window", 1840, 1990),
            ("Metal Window", 1990, 2140),
            ("Metal Window", 2140, 2290),
            ("Metal Window", 2290, 2440),
            ("CII Lines Window", 1330, 1340),
            ("Correlation Window", 1010, 1135),
        ]

        # Lopping over the windows
        for window, xmin, xmax in windows:

            # Defining the mask for the current window
            mask_window = (rest_grid >= xmin) & (rest_grid <= xmax)

            # If the current window contains the correlation window
            if window == "Correlation Window":

                # Compute masks for H₂ absorption features and synthetic profile
                mask_data, _, h2_synthetic_flux_rebinned = compute_h2_absorption_masks(rest_grid, 0.0, profile, np.zeros_like(rest_grid))
                # Overplot the synthetic profile used for the cross-correlation analysis
                plt.plot(rest_grid, continuum*h2_synthetic_flux_rebinned, alpha=0.5, color="b", label=rf"$H_{{2}}$ profile used for fitting")
                # Overplot absorption features used for the cross-correlation analysis
                plt.scatter(rest_grid[mask_data], continuum*h2_synthetic_flux_rebinned[mask_data], alpha=0.3, color="blue", s=10)
                # Overplot the associated complete synthetic profile for comparaison
                complete_profile = profile.get_complete_profile()
                _, _, h2_synthetic_flux_rebinned_complete = compute_h2_absorption_masks(rest_grid, 0.0, complete_profile, np.zeros_like(rest_grid))
                plt.plot(rest_grid, continuum*h2_synthetic_flux_rebinned_complete, alpha=0.5, color="green", label=rf"Complete $H_{{2}}$ profile")

                # Setting the corresponding y-axis range
                plt.ylim(np.nanmin(flux[mask_data])-0.15*continuum, np.nanmax(flux[mask_data])+0.08*continuum)
                # Updating the legend
                plt.legend(loc="lower right", fontsize=10)

            # Else, setting a personalized y-axis range for better visualisation of the metal lines
            else:
                # Computing the finite flux values in the current window
                finite_flux = flux[mask_window & np.isfinite(flux)]
                # If there are no finite flux values in the current window, skip to the next window to avoid errors when setting the y-axis limites
                if len(finite_flux) == 0:
                    continue
                # Establishing the y-axis limits with a margin of 10% of the flux range in the current window for better visualisation
                ymin = np.min(finite_flux)
                ymax = np.max(finite_flux)
                margin = 0.1 * (ymax - ymin)
                plt.ylim(ymin - margin, ymax + 2. * margin)

                # If the current window needs metal lines labelling
                if window not in ["Correlation Window", "CII Lines Window"]:
                    # Calling the function to plot the metal absorption lines labels on the plot
                    plot_metal_lines(ax, rest_grid, flux, xmin, xmax, ymin, ymax)
            
            # Setting the corresponding x-axis range for the current window
            plt.xlim(xmin, xmax)

            # Setting the plot title
            plt.title(f"{window}\n"+rf"{xmin}-{xmax} $\AA$")

            # Saving the figure
            output_file = os.path.join(output_folder, f"{"-".join(name.lower().split(" "))}/{xmin}-{xmax}_{"-".join(window.lower().split(" "))}.png")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, dpi=300)

            # Inform user
            tqdm.write(f"[INFO] {name} stacked spectrum saved in `{output_file}`")

        # Closing figure
        plt.close()

    # Return to the main function
    return

# Function to plot the metal lines labels on the stacked spectrum plot
def plot_metal_lines(ax : plt.Axes, wavelength : np.ndarray, flux : np.ndarray, xmin : float, xmax : float, ymin : float, ymax : float) -> None:
    """
    This function plots the metal lines labels on the stacked spectrum plot. 
    It retrieves the metal lines of interest from the line list of the VoigtFit 
    package, filters them based on the provided wavelength range, and then 
    annotates the plot with the corresponding labels.

    :param ax: The matplotlib Axes object on which to plot the metal lines labels.
    :type ax: plt.Axes
    :param wavelength: The wavelength array of the stacked spectrum.
    :type wavelength: np.ndarray
    :param flux: The flux array of the stacked spectrum.
    :type flux: np.ndarray
    :param xmin: The minimum wavelength of the current window.
    :type xmin: float
    :param xmax: The maximum wavelength of the current window.
    :type xmax: float
    :param ymin: The minimum flux value of the current window.
    :type ymin: float
    :param ymax: The maximum flux value of the current window.
    :type ymax: float
    :return: None
    """
    
    # List of the metal lines of interest for the analysis
    lines = [
        "SII_1250",   "SII_1253",  "SII_1259",  "SiII_1260", "CI_1277",
        "OI_1302",    "SiII_1304", "NiII_1317", "CI_1328",   "CII_1334", 
        "CIIa_1335.7","NiII_1370", "SiIV_1393", "SiIV_1402", "NiII_1454", 
        "SiII_1526",  "CIV_1548",  "CIV_1550",  "CI_1560",   "FeII_1608", 
        "FeII_1611",  "CI_1656",   "AlII_1670", "NiII_1709", "NiII_1741", 
        "NiII_1751",  "SiII_1808", "AlIII_1854","AlIII_1862","ZnII_2026", 
        "MgI_2026",   "CrII_2056", "ZnII_2062", "CrII_2062", "CrII_2066", 
        "FeII_2249",  "FeII_2260", "FeII_2344", "FeII_2374", "FeII_2382",
    ]

    # Initializing the dictionnary containing the name of the ion as key and the wavelength of the corresponding line as value for the lines of the ions of interest
    metal_lines = {}

    # Looping over the database of lines
    for row in lineList:
        # If the line transition is in the list of lines of interest and if the line wavelength is in the current window, adding it to the metal lines dictionnary for labelling on the plot
        if row["trans"] in lines:
            # If the line wavelength is in the current window, adding it to the metal lines dictionnary for labelling on the plot
            if xmin <= row["l0"] <= xmax:
                metal_lines[row["trans"]] = row["l0"]
    
    # Regrouping the lines by wavelength
    grouped = defaultdict(list)
    for label, wave in metal_lines.items():
        grouped[label.split("_")[1]].append((label, wave))
    
    # Initializing the dictionnary containing the merged metal lines
    merged_lines = {}
    # Looping over the grouped lines to merge the lines with a similar wavelength together
    for wave, items in grouped.items():
        # If it is a single line, keep it as it is
        if len(items) == 1:
            # Unpacking the label and wavelength of the line
            label, wave = items[0]
            # Adding the line to the merged lines dictionnary
            merged_lines[label] = wave
        # If there are multiple lines with a similar wavelength, merge them together
        else :
            # Retrieving the labels and wavelengths of the lines to merge
            labels = [item[0].split("_")[0] for item in items]
            waves = [item[1] for item in items]
            # Computing the mean wavelength of the lines to merge
            merged_wave = np.mean(waves)
            # Creating a merged label
            merged_label = " / ".join(sorted(set(labels))) + f"_{merged_wave:.1f}"
            # Adding the merged line to the dictionnary
            merged_lines[merged_label] = merged_wave

    # Looping over the metal lines retrieved to plot their labels on the spectrum
    for label, wave in merged_lines.items():
        # Computing the y position of the label using the flux array at the given wavelength
        mask = np.abs(wavelength - wave) <= 2.
        local = flux[mask & np.isfinite(flux)]
        y_base = np.nanmax(local) - .05*(ymax-ymin) if len(local) > 0 else ymax
        # Labelling the line with an arrow
        ax.annotate(label.split("_")[0], xy=(wave, y_base), xytext=(wave, y_base + .1*(ymax-ymin)), rotation=90, ha="center", va="bottom", fontsize=12, arrowprops=dict(arrowstyle="-", color="k", lw=.5))

    # Return to the main function
    return

# Function to create different stacked spectra from the analysis results
def spectra_stacker(mode : str = Modes.ALL, thresholds_dict : dict = {}) -> None:
    """
    This function creates the stacked spectra from the analysis results 
    using the provided mode and thresholds. It retrieves the analysis results, 
    loads the synthetic H₂ profiles, and computes the stacked spectra for 
    different stacking methods (raw, inverse variance weighted, mean, and 
    median). The resulting stacked spectra are then plotted and saved in 
    the specified output folder.

    :param mode: The mode of the analysis results to retrieve (default is `Modes.ALL`).
    :type mode: str
    :param thresholds_dict: A dictionary containing the thresholds for the analysis results (default is an empty dictionary).
    :type thresholds_dict: dict
    :return: None
    """

    # Loading cross-correlation analysis results
    AnalysisResults.load_results()
    # Loading synthetic H₂ profiles
    ProfileManager.load_all()

    # Retrieving the table
    table = AnalysisResults.results_survey(mode=mode, profile_name="all", thresholds_dict=thresholds_dict,)

    # Define folderpath using the mode and thresholds
    folderpath = f"{SPECTRA_PLOTS_FOLDER}stacked/{mode}"
    # Adding thresholds
    for key, value in thresholds_dict.items():
        # Skipping None values
        if value == (None, None):
            continue
        # Updating folderpath
        folderpath += f"_{COLUMN_FILE_LABELS[key]}-{value[0]}-{value[1]}"

    # Calling the function computing and plotting the stacked spectra from the selection
    plot_stacked_spectra(table, folderpath, profile_name="h2_profile_res-2650.0_ntot-20.0_J-0-1_Texc-75.0_b-3.0_pix-5.0")
