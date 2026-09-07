"""
This module contains the function related to the physical analysis 
(for physical interpretation of the results).

It contains the following functions:
- plot_categories_dist: plots the distribution of the correlation parameter for the different categories of systems that went through the visual inspection
- 
"""

# Package imports
from astropy.io import fits
from astropy.visualization import (ImageNormalize, PercentileInterval,)
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# Local imports
from src.desiqso.analysis.sample_statistics import bins
from src.desiqso.config import (settings, PHYSICAL_ANALYSIS_FOLDER, IMAGE_DATA_FOLDER)
from src.desiqso.constants import (Categories, ColNames, MagColNames, PhysColNames, Modes, COLUMN_FILE_LABELS,)
from src.desiqso.data.images import download_residual_images
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.utils.helpers import compute_legacy_magnitude

# Function to compute the distribution of the correlation parameter for the different categories of systems that went through the visual inspection
def plot_categories_dist() -> None:
    """
    This function plots the distribution of the correlation parameter for the 
    different categories of systems that went through the visual inspection.
    The distribution is plotted on the same figure with two different y-axis 
    scales to better visualize the distribution of the correlation parameter 
    for the "rejected" category that dominates the sample in terms of number of spectra.

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
    # Retrieving the maximum QSO redshift value in the results table of our candidates system to unsure the use of a relevent parente sample
    max_qsoz = results_table.loc[results_table[ColNames.CATEGORY].isin([Categories.CONFIRMED, Categories.UNSURE]), ColNames.QSO_Z].max() + .2
    # Retrieving the table corresponding to the parente sample
    parent_table = AnalysisResults.results_survey(mode=Modes.ALL, profile_name="best", thresholds_dict={ColNames.SNR : (3.0, None), ColNames.QSO_Z : (2.6, max_qsoz)})

    # Merging the magnitudes data table with the results table for easier handling of the data
    merged_table = parent_table[[ColNames.NAME, ColNames.CATEGORY] + x_cols].merge(mag_table, on=MagColNames.NAME, how="left")

    # Compute masks for the Legacy Survey band magnitudes to filter out the invalid values (i.e. null fluxes that lead to NaN magnitudes)
    g_mag_mask = merged_table[MagColNames.G_FLUX] > 0.001
    r_mag_mask = merged_table[MagColNames.R_FLUX] > 0.001
    z_mag_mask = merged_table[MagColNames.Z_FLUX] > 0.001
    # Final mask: all three magnitudes are valid
    mag_mask = g_mag_mask & r_mag_mask & z_mag_mask

    # Compute masks for the WISE band magnitudes to filter out the invalid values (i.e. null and negatives fluxes that lead to NaN magnitudes)
    w1_mag_mask = merged_table[MagColNames.W1_FLUX] > 0.001
    w2_mag_mask = merged_table[MagColNames.W2_FLUX] > 0.001
    w3_mag_mask = merged_table[MagColNames.W3_FLUX] > 0.001
    w4_mag_mask = merged_table[MagColNames.W4_FLUX] > 0.001
    # Final mask: all four magnitudes are valid
    w_mag_mask = w1_mag_mask & w2_mag_mask & w3_mag_mask & w4_mag_mask

    # Mask to access only the candidates system
    candidates_mask = (merged_table[ColNames.CATEGORY] == Categories.CONFIRMED) | (merged_table[ColNames.CATEGORY] == Categories.UNSURE)

    # Dictionnary of Milky Way transmission columns for the Legacy Survey and WISE bands to easily access them in the plotting loop
    mw_trans_dict = {
        MagColNames.G_FLUX : MagColNames.MW_TRANS_G,
        MagColNames.R_FLUX : MagColNames.MW_TRANS_R,
        MagColNames.Z_FLUX : MagColNames.MW_TRANS_Z,
        MagColNames.W1_FLUX : MagColNames.MW_TRANS_W1,
        MagColNames.W2_FLUX : MagColNames.MW_TRANS_W2,
        MagColNames.W3_FLUX : MagColNames.MW_TRANS_W3,
        MagColNames.W4_FLUX : MagColNames.MW_TRANS_W4,
    }
    
    # Updating the matplotlib settings
    settings["xtick.top"] = True
    plt.rcParams.update(**settings)
    # Switching off the interactive mode to avoid displaying the plots during the execution of the program
    plt.ioff()

    # ==============
    # Magnitude graphs
    # ============== 

    # List containing the band name, the flux column and the mask for each band to easily access them in the plotting loop
    flux_arrays = [
        ("Legacy Survey g", MagColNames.G_FLUX, mag_mask),
        ("Legacy Survey r", MagColNames.R_FLUX, mag_mask),
        ("Legacy Survey z", MagColNames.Z_FLUX, mag_mask),
        ("WISE W1", MagColNames.W1_FLUX, w_mag_mask),
        ("WISE W2", MagColNames.W2_FLUX, w_mag_mask),
        ("WISE W3", MagColNames.W3_FLUX, w_mag_mask),
        ("WISE W4", MagColNames.W4_FLUX, w_mag_mask),
    ]

    # Looping over the bands to plot the magnitude as a function of the QSO redshift
    for band, flux_col, mask in tqdm(flux_arrays, desc="Plotting magnitudes", unit="band"):

        # ==============
        # Magnitude VS QSO Redshift
        # ==============

        # Inform user

        # Computing the magnitudes for the parent sample and the candidates system using the fluxes columns and the mask provided
        mag_parent = compute_legacy_magnitude(merged_table.loc[mask, flux_col], merged_table.loc[mask, mw_trans_dict[flux_col]])
        mag_cand   = compute_legacy_magnitude(merged_table.loc[mask & candidates_mask, flux_col], merged_table.loc[mask & candidates_mask, mw_trans_dict[flux_col]])

        # Retrieving the QSO redshift values for the parent sample for the plot
        qsoz = merged_table.loc[mask, ColNames.QSO_Z]

        # Creating the figure
        _, _ = plt.subplots(figsize=(10, 6))

        # Plotting the magnitudes as a function of the QSO redshift for the parent sample and the candidates system
        plt.scatter(qsoz, mag_parent, s=2, label=f"parent sample (N={len(mag_parent)})", alpha=0.5, color="k", linewidth=0.5)
        plt.scatter(merged_table.loc[mask & candidates_mask, ColNames.QSO_Z], mag_cand, s=25, label=f"candidates (N={len(mag_cand)})", alpha=0.5, color="r", linewidth=0.5)

        # Retrieving the bins for the QSO redshift and computing the bins centers
        qsoz_bins = bins[ColNames.QSO_Z]
        bins_centers = (qsoz_bins[:-1] + qsoz_bins[1:]) * .5
        # Initializing array to store the median magnitude values for each bin
        median_mag = np.zeros_like(bins_centers)

        # Looping over the bins to compute the median magnitude value for each bin
        for i in range(len(bins_centers)):
            # Computing the mask for the current bin
            bin_mask = (qsoz >= qsoz_bins[i]) & (qsoz < qsoz_bins[i+1])
            # Computing the median magnitude value for the current bin and storing it in the array
            median_mag[i] = np.median(mag_parent[bin_mask]) if np.any(bin_mask) else np.nan
        # Plotting the median magnitude values as a function of the bins centers
        plt.plot(bins_centers, median_mag, label="median magnitude (parent sample)", color="blue", linewidth=2)

        # Setting the title, labels and legend
        plt.title(f"{band}-band magnitude as a function of {ColNames.QSO_Z}")
        plt.xlabel(ColNames.QSO_Z)
        plt.ylabel(f"{band}-band magnitude (mag)")
        plt.legend(title=ColNames.CATEGORY, loc="upper right", fontsize=12)
        # Adding a grid
        plt.grid(alpha=0.3)

        # Setting the x-axis and y-axis limits
        plt.ylim(np.quantile(mag_parent, 0.005), np.quantile(mag_parent, 0.995))
        plt.xlim(2.6, max_qsoz)

        # Defining the output file and ensuring that the folder exists
        outputfile = f"{PHYSICAL_ANALYSIS_FOLDER}mag-plots/{COLUMN_FILE_LABELS[ColNames.QSO_Z]}/{band.lower().split(" ")[-1]}-band-mag_vs_{COLUMN_FILE_LABELS[ColNames.QSO_Z]}.png"
        os.makedirs(os.path.dirname(outputfile), exist_ok=True)
        # Save figure in the dedicated folder
        plt.savefig(outputfile, dpi=400)

        # Inform user
        tqdm.write(f"[INFO] Magnitude in the {band}-band vs {ColNames.QSO_Z} saved in `{outputfile}`")

        # Closing the figure
        plt.close()

        # ==============
        # Magnitude excess distribution
        # ==============

        # Creating figure
        _, ax1 = plt.subplots(figsize=(10, 6))

        # Creating a twin axis to plot the color excess distribution for the parent sample and the candidates system on the same plot with different y-axis scales
        ax2 = ax1.twinx()

        # Creating a new DataFrame for data handling
        table = merged_table.loc[mask].copy()
        # Adding a column to the DataFrame corresponding to the current band magnitude
        table[band] = mag_parent
        # Adding a column to the DataFrame corresponding to the candidate mask
        table["is_candidate"] = candidates_mask
        # Adding a column corresponding to the x-axis bins
        table["x_bin"] = pd.cut(table[ColNames.QSO_Z], bins=qsoz_bins)
        # Computing the the mean color value for each bin
        bin_mean = table.groupby("x_bin")[band].transform("mean")
        # Adding a column to the DataFrame corresponding to the magnitude excess (i.e. the magnitude value minus the mean magnitude value for the corresponding x-axis bin)
        table["mag_excess"] = table[band] - bin_mean

        # Separating the magnitude excess values for the parent sample and the candidates system using the "is_candidate" column
        parent = table[~table["is_candidate"]]
        cand = table[table["is_candidate"]]

        # Defining plotting bins for proper binning using quantiles to avoid outliers to dominate the plot
        all_data = np.asarray(pd.concat([parent["mag_excess"], cand["mag_excess"]]))
        all_data = all_data[np.isfinite(all_data)]
        plotting_bins = np.linspace(np.quantile(all_data, 0.02), np.quantile(all_data, 0.98), 10)

        # Plotting the magnitude excess distribution as a function of the x-axis column for the parent sample
        ax1.hist(parent["mag_excess"], bins=plotting_bins, alpha=0.5, label=f"parent sample (N={len(parent)})", edgecolor="k", histtype='step', linewidth=2)
        # Plotting the magnitude excess distribution as a function of the x-axis column for the candidates system
        ax2.hist(cand["mag_excess"], bins=plotting_bins, alpha=0.7, label=f"candidates (N={len(cand)})", edgecolor="r", histtype='step', linewidth=2)

        # Adding a vertical line at the mean magnitude excess value for the candidates system and parent sample to highlight the difference between the two distributions
        ax2.axvline(cand["mag_excess"].median(), color="r", linestyle="--", label=f"median magnitude excess (candidates)", linewidth=2)
        ax1.axvline(parent["mag_excess"].median(), color="k", linestyle="--", label=f"median magnitude excess (parent sample)", linewidth=2)

        # Setting the title and axis labels
        plt.title(f"{band}-band magnitude excess as a function of {ColNames.QSO_Z}")
        ax1.set_xlabel(f"{band}-band magnitude excess (mag)")
        ax1.set_ylabel(f"Number of systems of the parent sample")
        ax2.set_ylabel(f"Number of systems of the candidates")
        # Setting the legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, title=ColNames.CATEGORY, loc="upper left", fontsize=10)
        # Adding a grid
        plt.grid(alpha=0.3)

        # Defining the output file and ensuring that the folder exists
        outputfile = f"{PHYSICAL_ANALYSIS_FOLDER}mag-plots/excesss-mag-dist/{band.lower().split(" ")[-1]}-band-excess-mag_dist-{COLUMN_FILE_LABELS[ColNames.QSO_Z]}.png"
        os.makedirs(os.path.dirname(outputfile), exist_ok=True)
        # Save figure in the dedicated folder
        plt.savefig(outputfile, dpi=400)

        # Inform user
        tqdm.write(f"[INFO] Excess magnitude distribution for the {band}-band saved in `{outputfile}`")

        # Closing the figure
        plt.close()

    # ==============
    # Color graphs
    # ==============

    # Configuring the colors arrays to plot with a list of tuples containing the color name, the color arrays and the mask
    colors_arrays = [
        ("g-r",   MagColNames.G_FLUX,  MagColNames.R_FLUX,  mag_mask),
        ("r-z",   MagColNames.R_FLUX,  MagColNames.Z_FLUX,  mag_mask),
        ("g-z",   MagColNames.G_FLUX,  MagColNames.Z_FLUX,  mag_mask),
        ("W1-W2", MagColNames.W1_FLUX, MagColNames.W2_FLUX, w_mag_mask),
        ("W2-W3", MagColNames.W2_FLUX, MagColNames.W3_FLUX, w_mag_mask),
        ("W3-W4", MagColNames.W3_FLUX, MagColNames.W4_FLUX, w_mag_mask),
        ("W1-W4", MagColNames.W1_FLUX, MagColNames.W4_FLUX, w_mag_mask),
        ("r-W1",  MagColNames.R_FLUX,  MagColNames.W1_FLUX, mag_mask & w_mag_mask),
        ("r-W2",  MagColNames.R_FLUX,  MagColNames.W2_FLUX, mag_mask & w_mag_mask),
        ("r-W3",  MagColNames.R_FLUX,  MagColNames.W3_FLUX, mag_mask & w_mag_mask),
        ("r-W4",  MagColNames.R_FLUX,  MagColNames.W4_FLUX, mag_mask & w_mag_mask),
        ("g-W1",  MagColNames.G_FLUX,  MagColNames.W1_FLUX, mag_mask & w_mag_mask),
        ("z-W1",  MagColNames.Z_FLUX,  MagColNames.W1_FLUX, mag_mask & w_mag_mask),
    ]

    # Dictionnary of color excess to keep for color excess correlation plotting
    parent_color_excess_dict : dict[str, pd.DataFrame] = {}
    cand_color_excess_dict : dict[str, pd.DataFrame] = {}

    # List of color excess correlation to plot
    color_excess_correlations = ["g-r", "r-W1", "W2-W3"]  

    # Looping over the columns to use as x-axis for the color graphs
    for x_col in x_cols:

        # Inform user
        print(f"\nPlotting the color graphs for x-axis: {x_col}...\n")

        # Looping over the colors to plot
        for color_name, flux_col1, flux_col2, mask in tqdm(colors_arrays, desc="Plotting colors", unit="color"):

            # ==============
            # Color VS QSO Redshift
            # ==============
            
            # Creating figure
            _, _ = plt.subplots(figsize=(10, 6))

            # Retrieving the x values for the parent sample and the candidates system using the masks defined above
            x = merged_table.loc[mask, x_col]
            x_cand = merged_table.loc[mask & candidates_mask, x_col]
            
            # Compute the magnitudes for the parent sample and the candidates system using the fluxes columns and the mask provided
            mag1      = compute_legacy_magnitude(merged_table.loc[mask, flux_col1], merged_table.loc[mask, mw_trans_dict[flux_col1]])
            mag2      = compute_legacy_magnitude(merged_table.loc[mask, flux_col2], merged_table.loc[mask, mw_trans_dict[flux_col2]])
            mag1_cand = compute_legacy_magnitude(merged_table.loc[mask & candidates_mask, flux_col1], merged_table.loc[mask & candidates_mask, mw_trans_dict[flux_col1]])
            mag2_cand = compute_legacy_magnitude(merged_table.loc[mask & candidates_mask, flux_col2], merged_table.loc[mask & candidates_mask, mw_trans_dict[flux_col2]])

            # Computing the color values for the parent sample and the candidates system
            color_array      = mag1      - mag2
            color_array_cand = mag1_cand - mag2_cand

            # Plotting the color values as a function of the x-axis column for the parent sample
            plt.scatter(x, color_array, s=2, label=f"parent sample (N={len(x)})", alpha=0.5, color="k", linewidth=0.5)
            # Plotting the color values as a function of the x-axis column for the candidates system
            plt.scatter(x_cand, color_array_cand, s=25, label=f"candidates (N={len(x_cand)})", alpha=0.7, color="r", linewidth=0.5)
            
            # Computing bins for the x-axis column
            x_bins = bins[x_col]
            # Computing bins centers
            bins_centers = (x_bins[:-1] + x_bins[1:]) * .5
            # Initializing array to store the median color values for each bin
            median_color = np.zeros_like(bins_centers)

            # Looping over the bins to compute the median color value for each bin
            for i in range(len(bins_centers)):
                # Computing the mask for the current bin
                bin_mask = (x >= x_bins[i]) & (x < x_bins[i+1])
                # Computing the median color value for the current bin and storing it in the array
                median_color[i] = np.median(color_array[bin_mask]) if np.any(bin_mask) else np.nan
            # Plotting the median color values as a function of the bins centers
            plt.plot(bins_centers, median_color, label="median color (parent sample)", color="blue", linewidth=2)

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
                plt.xlim(2.6, max_qsoz)

            # Defining the output file and ensuring that the folder exists
            outputfile = f"{PHYSICAL_ANALYSIS_FOLDER}color-plots/{COLUMN_FILE_LABELS[x_col]}/{color_name.lower()}_vs_{COLUMN_FILE_LABELS[x_col]}.png"
            os.makedirs(os.path.dirname(outputfile), exist_ok=True)
            # Save figure in the dedicated folder
            plt.savefig(outputfile, dpi=400)

            # Inform user
            tqdm.write(f"[INFO] Color plot for {color_name} vs {x_col} saved in `{outputfile}`")

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
            table = merged_table.loc[mask].copy()
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
            outputfile = f"{PHYSICAL_ANALYSIS_FOLDER}color-plots/excesss-color-dist/{color_name.lower()}-excess-color_dist.png"
            os.makedirs(os.path.dirname(outputfile), exist_ok=True)
            # Save figure in the dedicated folder
            plt.savefig(outputfile, dpi=400)

            # Inform user
            tqdm.write(f"[INFO] Excess color distribution for {color_name} saved in `{outputfile}`")

            # Closing the figure
            plt.close()

            # If the color will be used to plot excess color correlation
            if color_name in color_excess_correlations :
                # Save the color excess values for the parent sample and the candidates system
                parent_color_excess_dict[color_name] = parent[["color_excess", ColNames.NAME]]
                cand_color_excess_dict[color_name] = cand[["color_excess", ColNames.NAME]] 
        
        # ==============
        # Color excess correlations
        # ==============

        # Dictionnary of mask adapted to each band
        mag_masks = {
            "g" : mag_mask,
            "r" : mag_mask,
            "z" : mag_mask,
            "W1": w_mag_mask,
            "W2": w_mag_mask,
            "W3": w_mag_mask,
            "W4": w_mag_mask,
        }

        # Dictionnary of flux columns adapted to each band
        flux_cols = {
            "g" : MagColNames.G_FLUX,
            "r" : MagColNames.R_FLUX,
            "z" : MagColNames.Z_FLUX,
            "W1": MagColNames.W1_FLUX,
            "W2": MagColNames.W2_FLUX,
            "W3": MagColNames.W3_FLUX,
            "W4": MagColNames.W4_FLUX,
        }

        # Inform user
        print(f"\n[INFO] Plotting the color excess correlations for x-axis: {x_col}...")

        # Looping over the colors that will be used to plot the color excess correlation
        for color_name1 in color_excess_correlations :
            
            # Looping over the colors that are not the same as the one already selected
            for color_name2 in color_excess_correlations :
                if color_name1 != color_name2 :

                    # Inform user
                    print(f"\n[INFO] Plotting the correlation between {color_name1} and {color_name2}...")

                    # Retrieve the two bands for each of the colors
                    l_band_1, r_band_1 = color_name1.split("-")
                    l_band_2, r_band_2 = color_name2.split("-")

                    # Selecting the appropriate mask for each color
                    mask1 = mag_masks[l_band_1] & mag_masks[r_band_1]
                    mask2 = mag_masks[l_band_2] & mag_masks[r_band_2]
                    global_mask = mask1 & mask2

                    # Compute both color for both the parent sample and the candidates
                    par_color_1 = compute_legacy_magnitude(merged_table.loc[global_mask, flux_cols[l_band_1]], merged_table.loc[global_mask, mw_trans_dict[flux_cols[l_band_1]]]) - compute_legacy_magnitude(merged_table.loc[global_mask, flux_cols[r_band_1]], merged_table.loc[global_mask, mw_trans_dict[flux_cols[r_band_1]]])
                    par_color_2 = compute_legacy_magnitude(merged_table.loc[global_mask, flux_cols[l_band_2]], merged_table.loc[global_mask, mw_trans_dict[flux_cols[l_band_2]]]) - compute_legacy_magnitude(merged_table.loc[global_mask, flux_cols[r_band_2]], merged_table.loc[global_mask, mw_trans_dict[flux_cols[r_band_2]]])
                    
                    # Creating a new DataFrame for data handling
                    table = merged_table.loc[global_mask].copy()
                    # Adding a column to the DataFrame corresponding to each color
                    table[color_name1] = par_color_1
                    table[color_name2] = par_color_2
                    # Adding a column to the DataFrame corresponding to the candidate mask
                    table["is_candidate"] = candidates_mask
                    # Adding a column corresponding to the x-axis bins
                    table["x_bin"] = pd.cut(table[x_col], bins=x_bins)
                    # Computing the the mean color value for each bin and each color
                    bin_mean1 = table.groupby("x_bin")[color_name1].transform("mean")
                    bin_mean2 = table.groupby("x_bin")[color_name2].transform("mean")
                    # Adding a column to the DataFrame corresponding to the color excess (i.e. the color value minus the mean color value for the corresponding x-axis bin) for each color
                    table["color_excess1"] = table[color_name1] - bin_mean1
                    table["color_excess2"] = table[color_name2] - bin_mean2

                    # Separating the color excess values for the parent sample and the candidates system using the "is_candidate" column
                    parent = table[~table["is_candidate"]]
                    cand = table[table["is_candidate"]]

                    # Scattering the colors for the parent sample and the candidate systems
                    plt.scatter(parent["color_excess1"], parent["color_excess2"], s=1, label=f"parent sample (N={len(parent)})", alpha=0.5, color="k", linewidth=0.5)
                    plt.scatter(cand["color_excess1"], cand["color_excess2"], s=10, label=f"candidates (N={len(cand)})", alpha=0.7, color="r", linewidth=0.5)

                    # Setting the title, labels and legend
                    plt.title(f"{color_name1} excess color vs {color_name2} excess color")
                    plt.xlabel(f"{color_name1} excess color (mag)")
                    plt.ylabel(f"{color_name2} excess color (mag)")
                    plt.legend(title="Sample", loc="upper left", fontsize=12)
                    # Adding a grid
                    plt.grid(alpha=0.3)

                    # Defining the output file and ensuring that the folder exists
                    outputfile = f"{PHYSICAL_ANALYSIS_FOLDER}color-plots/excess-color-corr/{color_name1.lower()}-excess_vs_{color_name2.lower()}-excess.png"
                    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
                    # Save figure in the dedicated folder
                    plt.savefig(outputfile, dpi=400)
                    # Inform user
                    print(f"[INFO] Figure saved in {outputfile}")

                    # Closing the figure
                    plt.close()

    # Inform user
    print("\n[INFO] All the plots have been generated and saved in the dedicated folders.\n")

    # Return to the main function
    return 

# Function plotting the distribution of the chi-squared difference between two morphological models for the parent sample and the candidates system
def plot_morph_analysis() -> None:
    """
    This function plots the distribution of the chi-squared difference between two morphological 
    models (the morphological preference) for the parent sample and the candidates system.

    :return: None
    """

    # Loading the results
    AnalysisResults.load_results(verbose=True)

    # Retrieving the table containing the data about the physical properties of the systems
    phys_table = AnalysisResults._physical_data.copy()

    # Retrieving the table corresponding to the cross-correlation analysis results
    results_table = AnalysisResults.results_survey(mode=Modes.ALL, profile_name="best", thresholds_dict={ColNames.SNR : (3.0, None), ColNames.QSO_Z : (2.6, None)})
    # Retrieving the maximum QSO redshift value in the results table of our candidates system to unsure the use of a relevent parente sample
    max_qsoz = results_table.loc[results_table[ColNames.CATEGORY].isin([Categories.CONFIRMED, Categories.UNSURE]), ColNames.QSO_Z].max() + .2
    # Retrieving the table corresponding to the parente sample
    parent_table = AnalysisResults.results_survey(mode=Modes.ALL, profile_name="best", thresholds_dict={ColNames.SNR : (3.0, None), ColNames.QSO_Z : (2.6, max_qsoz)})

    # Merging the magnitudes data table with the results table for easier handling of the data
    merged_table = parent_table[[ColNames.NAME, ColNames.CATEGORY]].merge(phys_table, on=PhysColNames.NAME, how="left")

    # Mask to access only the candidates system
    candidates_mask = (merged_table[ColNames.CATEGORY] == Categories.CONFIRMED) | (merged_table[ColNames.CATEGORY] == Categories.UNSURE)

    # Updating the matplotlib settings
    settings["xtick.top"] = True
    plt.rcParams.update(**settings)
    # Switching off the interactive mode to avoid displaying the plots during the execution of the program
    plt.ioff()

    # ==============
    # Difference in chi-squared for two models distribution (morphological preference)
    # ==============

    # Inform user
    print("\n[INFO] Plotting the morphological preference for the parent sample and the candidates system...")

    # List of the data to plot
    data_to_plot = [
        ("DEV-PSF", PhysColNames.DCHISQ_DEV, PhysColNames.DCHISQ_PSF),
        ("EXP-PSF", PhysColNames.DCHISQ_EXP, PhysColNames.DCHISQ_PSF),
        ("REX-PSF", PhysColNames.DCHISQ_REX, PhysColNames.DCHISQ_PSF),
        ("SER-PSF", PhysColNames.DCHISQ_SER, PhysColNames.DCHISQ_PSF),
        ("DEV-REX", PhysColNames.DCHISQ_DEV, PhysColNames.DCHISQ_REX),
        ("EXP-REX", PhysColNames.DCHISQ_EXP, PhysColNames.DCHISQ_REX),
        ("SER-REX", PhysColNames.DCHISQ_SER, PhysColNames.DCHISQ_REX),
        ("DEV-EXP", PhysColNames.DCHISQ_DEV, PhysColNames.DCHISQ_EXP),
        ("SER-EXP", PhysColNames.DCHISQ_SER, PhysColNames.DCHISQ_EXP),
    ]

    # Looping over the data to plot
    for morph_pref, col1, col2 in tqdm(data_to_plot, desc="Plotting morphological preference", unit="plot"):

        # Creating a mask to filter out the invalid values (i.e. null values and non finite values)
        mask = np.isfinite(merged_table[col1]) & np.isfinite(merged_table[col2]) & (merged_table[col1] != 0) & (merged_table[col2] != 0)
        # Keeping only a subset of the table containing the data to plot for the parent sample and the candidates system
        parent_data = merged_table.loc[mask].copy()
        cand_data = merged_table.loc[mask & candidates_mask].copy()

        # Computing the morphological preference for each table
        parent_data[morph_pref] = parent_data[col1] - parent_data[col2]
        cand_data[morph_pref] = cand_data[col1] - cand_data[col2]

        # Defining the bins for the histogram using quantiles
        vmin, vmax = np.percentile(np.concatenate([parent_data[morph_pref], cand_data[morph_pref]]),[0.1, 99.9])
        bins = np.linspace(vmin, vmax, 50)

        # Creating the figure
        _, ax = plt.subplots(figsize=(10, 6))

        # Plotting a stacked histogram of the parent sample and the candidates system
        ax.hist([cand_data[morph_pref], parent_data[morph_pref]], bins=bins, stacked=True, histtype="stepfilled", edgecolor="black", facecolor=["g","b"], alpha=0.7, linewidth=1.2, label=[f"candidates (N={len(cand_data)})", f"parent sample (N={len(parent_data)})"])

        # Setting the title, labels and legend
        plt.title(f"Morphological preference: {morph_pref}")
        plt.xlabel(rf"$\Delta \chi^2_{{{morph_pref}}}$")
        plt.ylabel(f"Number of systems (log)")
        plt.legend(title="Samples", loc="upper right", fontsize=12)
        
        # Adding a grid
        plt.grid(alpha=0.3)

        # Setting the y-axis scale
        plt.yscale("log")

        # Defining the output file and ensuring that the folder exists
        outputfile = f"{PHYSICAL_ANALYSIS_FOLDER}morph-pref/{morph_pref}_morph-pref.png"
        os.makedirs(os.path.dirname(outputfile), exist_ok=True)
        # Save figure in the dedicated folder
        plt.savefig(outputfile, dpi=400)

        # Inform user
        tqdm.write(f"[INFO] Morphological preference plot {morph_pref} saved in `{outputfile}`")

        # Closing the figure
        plt.close()

    # Return to the main function
    return

# Function to perform the analysis of the residual images
def residual_image_stacking_analysis(mode : str = Modes.PARENT, thresholds_dict : dict = {}) -> None:
    """
    This function performs the analysis of the residual images by stacking them and plotting the 
    results for the parent sample and the candidates system. The stacking is performed using three 
    different methods: raw stacking, mean stacking and median stacking.

    :param mode: The mode of the analysis, either "Modes.PARENT" or "Modes.CANDIDATES". Default is "Modes.PARENT".
    :type mode: str
    :param thresholds_dict: A dictionary containing the thresholds for the analysis. Default is an empty dictionary.
    :type thresholds_dict: dict
    :return: None
    """

    # Loading the results
    AnalysisResults.load_results(verbose=True)

    # Ensuring that the images folder exists
    os.makedirs(IMAGE_DATA_FOLDER, exist_ok=True)

    # Retrieving the table corresponding to the spectra of requested sample
    table = AnalysisResults.results_survey(mode=mode, profile_name="best", thresholds_dict=thresholds_dict)

    # Excluding the systems for which the image is already downloaded
    downloaded = [name[:-5] + ".fits" for name in os.listdir(IMAGE_DATA_FOLDER)]
    download_table = table[~table[ColNames.FILENAME].isin(downloaded)].copy()

    # Download the images for the remaining systems
    download_residual_images(download_table)

    # Initializing the images list
    images = []
    # Looping over the files in the table to stack
    for file in table[ColNames.FILENAME]:
        # Appending the image to the list
        images.append(fits.getdata(os.path.join(IMAGE_DATA_FOLDER, file)))
    
    # Converting the list to an array for easier handling of the stacks
    cube = np.array(images)

    # Creating the stacks
    stack_raw = np.sum(cube, axis=0)
    stack_mean = np.mean(cube, axis=0)
    stack_median = np.median(cube, axis=0)

    # Defining the list of the bands to use
    bands = ["g", "r", "i", "z"]

    # Updating the matplotlib settings
    settings["xtick.top"] = True
    plt.rcParams.update(**settings)
    # Switching off the interactive mode to avoid displaying the plots during the execution of the program
    plt.ioff()

    # Informing user
    print("\n[INFO] Plotting stacked residual images...\n")

    # Looping over the stacks
    for label, stack in tqdm([("Raw", stack_raw), ("Mean", stack_mean), ("Median", stack_median)], desc="Plotting stacked residual images", unit="stack"):
        # Looping over the bands
        for i, band in enumerate(bands):

            # Retrieving the image of the band from the stack
            image = stack[i]

            # Normalizing the image using the dedicated `astropy` class
            norm = ImageNormalize(image, interval=PercentileInterval(99.5))

            # Creating the figure
            _, _ = plt.subplots(figsize=(10, 6))

            # Plotting the image
            plt.imshow(image, origin="lower", norm=norm, cmap="viridis")
            # Adding the colorbar
            plt.colorbar(label="Residual stacked flux")
            # Adding a title
            plt.title(f"Residual {label.lower()} stacked image ({band})")
            # Adding a grid
            plt.grid(alpha=0.3)

            # Defining the output file and ensuring that the folder exists
            outputfile = f"{PHYSICAL_ANALYSIS_FOLDER}residual-stacking/{mode.lower()}/{band}-band_{label.lower()}-residual-stacking.png"
            os.makedirs(os.path.dirname(outputfile), exist_ok=True)
            # Save figure in the dedicated folder
            plt.savefig(outputfile, dpi=400)

            # Inform user
            tqdm.write(f"[INFO] Residual {label.lower()} stacked image ({band}) saved in `{outputfile}`")

            # Closing the figure
            plt.close()
        
    # Inform user
    print("\n[INFO] Stacked residual images successfully plotted.\n")

    # Return to the main function
    return
