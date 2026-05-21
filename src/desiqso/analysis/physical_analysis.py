"""
This module contains the function related to the physical analysis (for physical interpretation of the results).
"""

# Package imports
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Local imports
from src.desiqso.analysis.sample_statistics import bins
from src.desiqso.config import (settings, PHYSICAL_ANALYSIS_FOLDER,)
from src.desiqso.constants import (Categories, ColNames, MagColNames, Modes, COLUMN_FILE_LABELS,)
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.utils.helpers import compute_legacy_magnitude

# Function to compute the distribution of the correlation parameter for the different categories of systems that went through the visual inspection
def plot_categories_dist() -> None:
    """
    This function plots the distribution of the correlation parameter for the 
    different categories of systems that went through the visual inspection.
    The distribution is plotted on the same figure with two different y-axis 
    scales to better visualize the distribution of the correlation parameter 
    for the REJECTED category that dominates the sample in terms of number of spectra.

    :param None: This function does not take any input parameters.
    :return None: This function does not return anything, it just saves the plot in the dedicated folder.
    """

    # Load the results
    AnalysisResults.load_results(verbose=False)

    # Retrieving the table corresponding to the spectra that went through the visual inspection
    visual_table = AnalysisResults.results_survey(mode=Modes.VISUAL, profile_name="best", thresholds_dict={})

    # Updating the matplotlib settings
    settings["xtick.top"] = True
    plt.rcParams.update(**settings)
    # Switching off the interactive mode to avoid displaying the plots during the execution of the program
    plt.ioff()

    # Inform user
    print("\n[INFO] Plotting the evolution of the fraction of each category with the correlation parameter...\n")

    # Defining the bins in which compute the histogram
    bins = np.arange(0.42, 1.0, 0.02)
    
    # Creating figure
    _, ax1 = plt.subplots(figsize=(10, 6))

    # Creating secondary axis
    ax2 = ax1.twinx()

    # Defining the colors for each category
    colors = {
        Categories.CONFIRMED:    "green", 
        Categories.UNSURE   :    "orange", 
        Categories.REJECTED :    "red", 
    }

    # Looping over the the categories
    for category, group in visual_table.groupby(ColNames.CATEGORY):

        # Selecting the axis to plot the histogram on depending on the category
        ax = ax1 if category == Categories.REJECTED else ax2
        # Plotting the histogram
        ax.hist(group[ColNames.CORR_PARAM], bins=bins, histtype='step', linewidth=2, label=f"{category}, N={len(group)}", edgecolor=colors[category])

    # Setting the title, labels and legend
    ax1.set_title("Distribution of the correlation parameter", fontsize=16)
    ax1.set_xlabel("Correlation parameter", fontsize=14)
    ax1.set_ylabel("Number of spectra (REJECTED)", fontsize=14)
    ax2.set_ylabel("Number of spectra (CONFIRMED / UNSURE)", fontsize=14)
    
    # Merge legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles2 + handles1, labels2 + labels1, title=ColNames.CATEGORY, loc="upper right", fontsize=12)

    # Defining the output file and ensuring that the folder exists
    outputfile = f"{PHYSICAL_ANALYSIS_FOLDER}corr-param_distribution_categories.png"
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    # Save figure in the dedicated folder
    plt.savefig(outputfile, dpi=400)
    # Closing the figure to free up memory
    plt.close()

    # Inform user
    print(f"[INFO] Plot saved in {outputfile}\n")

    # Return to the main function
    return

# Function to plot the color of the systems as a function of different parameters and the color excess distribution for the parent sample and the candidates system
def plot_color_graphs(x_cols : list) -> None:
    """
    This function plots the color of the systems as a function of different parameters 
    and the color excess distribution for the parent sample and the candidates system.
    The color excess is defined as the difference between the color of each system and 
    the mean color of the parent sample for the corresponding x-axis parameter bin.

    :param x_cols: List of column names to use as x-axis for the color graphs.
    :type x_cols: list
    :return None: This function does not return anything, it just saves the plots in the dedicated folder.
    """

    # Loading the results
    AnalysisResults.load_results(verbose=True)

    # Retrieving the table corresponding to the magnitude data
    mag_table = AnalysisResults._magnitudes.copy()
    # Retrieving the table corresponding to the cross-correlation analysis results
    results_table = AnalysisResults.results_survey(mode=Modes.ALL, profile_name="best", thresholds_dict={ColNames.SNR : (3.0, None), ColNames.QSO_Z : (2.6, None)})
    # Merging the magnitudes data table with the results table for easier handling of the data
    merged_table = results_table[[ColNames.NAME, ColNames.CATEGORY] + x_cols].merge(mag_table, on=MagColNames.NAME, how="left")

    # Compute masks for the magnitudes to filter out the invalid values (i.e. null fluxes that lead to NaN magnitudes)
    g_mag_mask = merged_table[MagColNames.G_FLUX] > 0.001
    r_mag_mask = merged_table[MagColNames.R_FLUX] > 0.001
    z_mag_mask = merged_table[MagColNames.Z_FLUX] > 0.001
    # Final mask: all three magnitudes are valid
    mag_mask = g_mag_mask & r_mag_mask & z_mag_mask

    # Mask to access only the candidates system
    candidates_mask = (merged_table[ColNames.CATEGORY] == Categories.CONFIRMED) | (merged_table[ColNames.CATEGORY] == Categories.UNSURE)

    # Compute magnitudes from fluxes for the parent sample (i.e. all spectra that have valid magnitudes)
    g_mag = compute_legacy_magnitude(merged_table.loc[mag_mask, MagColNames.G_FLUX], merged_table.loc[mag_mask, MagColNames.MW_TRANS_G])
    r_mag = compute_legacy_magnitude(merged_table.loc[mag_mask, MagColNames.R_FLUX], merged_table.loc[mag_mask, MagColNames.MW_TRANS_R])
    z_mag = compute_legacy_magnitude(merged_table.loc[mag_mask, MagColNames.Z_FLUX], merged_table.loc[mag_mask, MagColNames.MW_TRANS_Z])
    # Compute magnitudes from fluxes for the candidates system
    g_mag_cand = compute_legacy_magnitude(merged_table.loc[mag_mask & candidates_mask, MagColNames.G_FLUX], merged_table.loc[mag_mask & candidates_mask, MagColNames.MW_TRANS_G])
    r_mag_cand = compute_legacy_magnitude(merged_table.loc[mag_mask & candidates_mask, MagColNames.R_FLUX], merged_table.loc[mag_mask & candidates_mask, MagColNames.MW_TRANS_R])
    z_mag_cand = compute_legacy_magnitude(merged_table.loc[mag_mask & candidates_mask, MagColNames.Z_FLUX], merged_table.loc[mag_mask & candidates_mask, MagColNames.MW_TRANS_Z])

    # Configuring the colors arrays to plot with a dictionary
    colors_arrays = [
        ("g-r", g_mag - r_mag, g_mag_cand - r_mag_cand),
        ("r-z", r_mag - z_mag, r_mag_cand - z_mag_cand),
        ("g-z", g_mag - z_mag, g_mag_cand - z_mag_cand),
    ]
    
    # Updating the matplotlib settings
    settings["xtick.top"] = True
    plt.rcParams.update(**settings)
    # Switching off the interactive mode to avoid displaying the plots during the execution of the program
    plt.ioff()

    # Looping over the columns to use as x-axis for the color graphs
    for x_col in x_cols:

        # Looping over the colors to plot
        for color_name, color_array, color_array_cand in colors_arrays:

            # ==============
            # Plot
            # ==============
            
            # Creating figure
            _, _ = plt.subplots(figsize=(10, 6))

            # Retrieving the x values for the parent sample and the candidates system using the masks defined above
            x = merged_table.loc[mag_mask, x_col]
            x_cand = merged_table.loc[mag_mask & candidates_mask, x_col]

            # Plotting the color values as a function of the x-axis column for the parent sample
            plt.scatter(x, color_array, color_array, label=f"parent sample (N={len(x)})", alpha=0.5, color="k", linewidth=0.5)
            # Plotting the color values as a function of the x-axis column for the candidates system
            plt.scatter(x_cand, color_array_cand, label=f"candidates (N={len(x_cand)})", alpha=0.7, color="r", linewidth=0.5)

            # Computing bins for the x-axis column
            x_bins = bins[x_col]
            # Computing bins centers
            bins_centers = (x_bins[:-1] + x_bins[1:]) * .5
            # Initializing array to store the mean color values for each bin
            mean_color = np.zeros_like(bins_centers)

            # Looping over the bins to compute the mean color value for each bin
            for i in range(len(bins_centers)):
                # Computing the mask for the current bin
                bin_mask = (x >= x_bins[i]) & (x < x_bins[i+1])
                # Computing the mean color value for the current bin and storing it in the array
                mean_color[i] = np.median(color_array[bin_mask]) if np.any(bin_mask) else np.nan
            # Plotting the mean color values as a function of the bins centers
            plt.plot(bins_centers, mean_color, label="median color (parent sample)", color="blue", linewidth=2)

            # Setting the title, labels and legend
            plt.title(f"{color_name} color as a function of {x_col}")
            plt.xlabel(x_col)
            plt.ylabel(f"{color_name} color (mag)")
            plt.legend(title=ColNames.CATEGORY, loc="upper left", fontsize=12)
            # Adding a grid
            plt.grid(alpha=0.3)

            # Setting the x-axis and y-axis limits
            if color_name == "r-z":
                plt.ylim(0.0, np.quantile(color_array, 0.995))
            else:
                plt.ylim(np.quantile(color_array, 0.005), np.quantile(color_array, 0.995))
            if x_col == ColNames.QSO_Z:
                plt.xlim(2.5, 5.0)

            # Defining the output file and ensuring that the folder exists
            outputfile = f"{PHYSICAL_ANALYSIS_FOLDER}color-plots/{color_name}_vs_{COLUMN_FILE_LABELS[x_col]}.png"
            os.makedirs(os.path.dirname(outputfile), exist_ok=True)
            # Save figure in the dedicated folder
            plt.savefig(outputfile, dpi=300)

            # Closing the figure
            plt.close()

            # ==============
            # Color excess distribution
            # ==============

            # Creating figure
            _, ax1 = plt.subplots(figsize=(10, 6))

            # Creating a twin axis to plot the color excess distribution for the parent sample and the candidates system on the same plot with different y-axis scales
            ax2 = ax1.twinx()

            # Creating a new DataFrame for data handling
            table = merged_table.copy()
            # Adding a column to the DataFrame corresponding to the current color
            table[color_name] = color_array
            # Adding a column to the DataFrame corresponding to the candidate mask
            table["is_candidate"] = candidates_mask
            # Adding a column corresponding to the x-axis bins
            table["x_bin"] = pd.cut(table[x_col], bins=x_bins)
            # Computing the the mean color value for each bin
            bin_mean = table.groupby("x_bin")[color_name].transform("mean")
            # Adding a column to the DataFrame corresponding to the color excess (i.e. the color value minus the mean color value for the corresponding x-axis bin)
            table["color_excess"] = table[color_name] - bin_mean

            # Separating the color excess values for the parent sample and the candidates system using the "is_candidate" column
            parent = table[~table["is_candidate"]]
            cand = table[table["is_candidate"]]

            # Defining plotting bins for proper binning using quantiles to avoid outliers to dominate the plot
            all_data = np.asarray(pd.concat([parent["color_excess"], cand["color_excess"]]))
            all_data = all_data[np.isfinite(all_data)]
            plotting_bins = np.linspace(np.quantile(all_data, 0.02), np.quantile(all_data, 0.98), 10)

            # Plotting the color excess distribution as a function of the x-axis column for the parent sample
            ax1.hist(parent["color_excess"], bins=plotting_bins, alpha=0.5, label=f"parent sample (N={len(parent)})", edgecolor="k", histtype='step', linewidth=2)
            # Plotting the color excess distribution as a function of the x-axis column for the candidates system
            ax2.hist(cand["color_excess"], bins=plotting_bins, alpha=0.7, label=f"candidates (N={len(cand)})", edgecolor="r", histtype='step', linewidth=2)

            # Adding a vertical line at the mean color excess value for the candidates system and parent sample to highlight the difference between the two distributions
            ax2.axvline(cand["color_excess"].median(), color="r", linestyle="--", label=f"median color excess (candidates)", linewidth=2)
            ax1.axvline(parent["color_excess"].median(), color="k", linestyle="--", label=f"median color excess (parent sample)", linewidth=2)

            # Setting the title and axis labels
            plt.title(f"{color_name} color excess as a function of {x_col}")
            ax1.set_xlabel(f"{color_name} color excess (mag)")
            ax1.set_ylabel(f"Number of systems of the parent sample")
            ax2.set_ylabel(f"Number of systems of the candidates")
            # Setting the legend
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(handles1 + handles2, labels1 + labels2, title=ColNames.CATEGORY, loc="upper left", fontsize=10)
            # Adding a grid
            plt.grid(alpha=0.3)

            # Defining the output file and ensuring that the folder exists
            outputfile = f"{PHYSICAL_ANALYSIS_FOLDER}color-plots/{color_name}_vs_dist-{COLUMN_FILE_LABELS[x_col]}.png"
            os.makedirs(os.path.dirname(outputfile), exist_ok=True)
            # Save figure in the dedicated folder
            plt.savefig(outputfile, dpi=300)

            # Closing the figure
            plt.close()

    # Return to the main function
    return 
