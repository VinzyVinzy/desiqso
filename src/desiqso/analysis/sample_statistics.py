"""
This module contains the functions to plot the initial sample statistics for each category of spectra, 
using the associated colors for each category (confirmed, borderline, rejected and other), and displaying 
the number of spectra used for each category in the plot. It also adds a vertical line to indicate the t
hreshold for the correlation parameter if the column plotted is the correlation parameter.
"""

# Packages import
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# Local imports
from src.desiqso.config import (settings, CORRELATION_PARAM_THRESHOLD, SAMPLE_STATISTICS_FOLDER)
from src.desiqso.constants import (ColNames, COLUMN_FILE_LABELS, Modes,)
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.models.profile import ProfileManager
from src.desiqso.utils.helpers import compute_column_weights

# ================================
# Global variables for plots configuration
# ================================

# Defining the colors for each category
colors = {"confirmed":"green", "borderline":"orange", "rejected":"red", "other":"blue"}

# Defining the bins for the histograms of each column
bins = {
    ColNames.SNR        : np.arange(0, 50.25, 0.25),
    ColNames.CONTINUUM  : np.arange(0, 10.5, 0.5),
    ColNames.QSO_Z      : np.arange(2.5, 6.05, 0.05),
    ColNames.Z          : np.arange(2.5, 6.05, 0.05),
    ColNames.RA         : np.arange(0, 370, 10),
    ColNames.DEC        : np.arange(-40, 95, 5),
    ColNames.CORR_PARAM : np.arange(0., 1.025, 0.025),
    ColNames.CORE_TRANS : np.arange(-1., 1., 0.05),
    ColNames.GRADE      : np.arange(-0.5, 6.5, 1),
    ColNames.REL_SPEED  : np.arange(-2600, 2600, 100),
}

# Definig the y positions for the text displaying the number of spectra for each category in the histograms
y_pos_dict = {"other" : 0.95, "confirmed" : 0.90, "borderline" : 0.85, "rejected" : 0.80}



# Function to plot the initial sample statistics
def plot_sample_statistics() -> None:
    """
    This function plots the initial sample statistics for each category of spectra, using the 
    associated colors for each category (confirmed, borderline, rejected and other), and 
    displaying the number of spectra used for each category in the plot. It also adds a vertical 
    line to indicate the threshold for the correlation parameter if the column plotted is the 
    correlation parameter.
    """

    # ================================
    # Configuration and data loading
    # ================================

    # Loading cross-correlation analysis results
    AnalysisResults.load_results()
    # Loading all the synthetic profiles
    ProfileManager.load_all()

    # Updating the matplotlib settings
    plt.rcParams.update(**settings)
    # Switching off the interactive mode to avoid displaying the plots during the execution of the program
    plt.ioff()

    # Inform user
    print("\n[INFO] Retrieving the tables to plot...\n")

    # Retrieving the tables for each category of spectra
    low_snr_table = AnalysisResults._low_snr.copy()
    failed_table  = AnalysisResults._failed.copy()
    # Retrieving the tables for the successful, valid, confirmed 
    successful_table = AnalysisResults.results_survey(Modes.ALL, "best", {})
    valid_table      = AnalysisResults.results_survey(Modes.VALID, "best", {})
    confirmed_table  = AnalysisResults.results_survey(Modes.CONFIRMED, "best", {})
    rejected_table   = AnalysisResults.results_survey(Modes.REJECTED, "best", {})
    new_table        = AnalysisResults.results_survey(Modes.NEW, "best", {})


    # Defining the dictionary of the tables to plot for each category of spectra
    table_dict = {
        "Low SNR"        : low_snr_table,
        "Failed"         : failed_table,
        "Successful"     : successful_table,
        "Valid"          : valid_table,
        "Confirmed"      : confirmed_table,
        "Rejected"       : rejected_table,
        "SNR Range"      : successful_table,
        "New Candidates" : new_table,
    }

    # Defining the dictionary of the columns to plot for each category of spectra
    cols_dict = {
        "Low SNR"       : [ColNames.SNR, ColNames.CONTINUUM, ColNames.QSO_Z,             ColNames.RA, ColNames.DEC,],
        "Failed"        : [ColNames.SNR, ColNames.CONTINUUM, ColNames.QSO_Z,             ColNames.RA, ColNames.DEC,],
        "Successful"    : [ColNames.SNR, ColNames.CONTINUUM, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Confirmed"     : [ColNames.SNR, ColNames.CONTINUUM, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Rejected"      : [ColNames.SNR, ColNames.CONTINUUM, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Valid"         : [ColNames.SNR, ColNames.CONTINUUM, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "SNR Range"     : [ColNames.SNR,                     ColNames.QSO_Z,             ColNames.RA, ColNames.DEC, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE,],
        "New Candidates": [ColNames.SNR, ColNames.CONTINUUM, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
    }

    # ================================
    # Plotting the histograms for each category of spectra and column
    # ================================

    # Looping over the categories of spectra and columns to plot and plotting the histogram for each category and column
    for key, table in tqdm(table_dict.items(), desc="Plotting sample statistics", unit="category"):

        # Looping over the columns to plot for the category of spectra
        for col in cols_dict[key]:
            
            # Creating the figure and axis
            _, ax = plt.subplots(figsize=(12,8))

            # Matching the category of spectra to plot with the associated plotting function
            match key:
                
                # For the successful and valid spectra, plotting the histogram with the associated categories colors
                case "Successful" | "Valid":
                    # Plotting the histogram with the associated categories colors
                    plot_sample_statistics_categories(table, col, ax)

                # For the spectra with low SNR , plotting the histogram with the associated threshold for the SNR
                case "SNR Range":
                    # Plotting the histogram with the associated function
                    plot_sample_statistics_threshold(table, col, 15., ColNames.SNR, ax)

                # For all the other categories
                case _:
                    # Plotting the histogram with the associated function
                    plot_sample_statistics_standard(table, col, ax)
            
            # Setting the labels and title of the plot
            plt.xlabel(col)
            plt.ylabel("Percentage of total spectra")
            plt.title(f"Distribution of {col} for {key} spectra")

            # Setting the x-axis limits to better visualize the distribution
            plt.xlim(bins[col][0], bins[col][-1])

            # Setting the y-axis to display the percentage instead of the fraction
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

            # Adding a vertical line to indicate the threshold for the correlation parameter if the column is the correlation parameter
            if col == ColNames.CORR_PARAM:
                ax.axvline(x=CORRELATION_PARAM_THRESHOLD, color="red", linestyle="--", linewidth=2)
                        
            # Adding a grid to the plot
            ax.grid(True, alpha=.5)

            # Saving the plot in the output directory, creating the directory if it does not exist
            output_dir = os.path.join(SAMPLE_STATISTICS_FOLDER, "_".join(key.lower().split(" ")))
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{COLUMN_FILE_LABELS[col]}_distribution.png"), dpi=300)

            # Closing the plot to avoid displaying it during the execution of the program
            plt.close()
    
    # Inform user
    print(f"\n[INFO] Sample statistics plots saved in the `{SAMPLE_STATISTICS_FOLDER}` directory.\n")

# Function to plot the histogram of a given column for each category of spectra in the given table
def plot_sample_statistics_categories(table: pd.DataFrame, col: str, ax: plt.Axes) -> None:
    """
    This function plots the histogram of the given column for each category of spectra in 
    the given table, using the associated colors for each category and displaying the number 
    of spectra used for each category in the plot.

    :param table: The table containing the spectra data to plot, with a column for the category
    of each spectrum and a column for the values to plot.
    :type table: pd.DataFrame
    :param col: The name of the column to plot the histogram for.
    :type col: str
    :param ax: The `matplotlib` axis to plot the histogram on.
    :type ax: plt.Axes
    :return: None
    """

    # Looping over the categories and plotting the histogram for each category with the associated color
    for category in table[ColNames.CATEGORY].unique():
        # Selecting the category and column to plot, dropping the NaN values
        category_table : pd.DataFrame = table[table[ColNames.CATEGORY] == category].dropna(subset=[col])
        # Plotting the histogram for the category and column with the associated color and label, using the weights to plot the fraction of spectra in each bin instead of the count
        category_table[col].hist(bins=bins[col], weights=compute_column_weights(category_table, col), ax=ax, label=category, edgecolor=colors[category], histtype='step', linewidth=2)
        # Displaying the number of spectra used for the category and column
        ax.text(0.85, y_pos_dict[category], rf"N$_{{{category}}}$ = {len(category_table)}", transform=ax.transAxes, fontsize=12, va='top')
    
    # Adding the legend to the plot
    ax.legend(title=ColNames.CATEGORY, loc="upper left")

    # Returning to the main function
    return

# Function to plot the histogram of a given column for two slides of the sample defined by a threshold on a given column
def plot_sample_statistics_threshold(table : pd.DataFrame, col : str, threshold : float, threshold_col : str, ax : plt.Axes) -> None:
    """
    This function plots the histogram of the given column for two slides of the sample defined by a threshold on a given column, 
    using different colors for each slide and displaying the number of spectra used for each slide in the plot.

    :param table: The table containing the spectra data to plot, with a column for the category
    of each spectrum and a column for the values to plot.
    :type table: pd.DataFrame
    :param col: The name of the column to plot the histogram for.
    :type col: str
    :param threshold: The value of the threshold to define the two slides of the sample.
    :type threshold: float
    :param threshold_col: The name of the column to apply the threshold on.
    :type threshold_col: str
    :param ax: The `matplotlib` axis to plot the histogram on.
    :type ax: plt.Axes
    :return: None
    """

    # Selecting the part of the table satisfying the condition on the threshold column and dropping the NaN values for the column to plot
    table_low = table[table[threshold_col] < threshold].dropna(subset=[col])
    # Plotting the histogram for the part of the table selected
    table_low[col].hist(bins=bins[col], weights=compute_column_weights(table_low, col), ax=ax, edgecolor="blue", label=rf"{threshold_col} < {threshold}", histtype="step", linewidth=2)
    # Displaying the number of spectra used for the part of the table selected
    ax.text(0.80, 0.95,rf"$N_{{{threshold_col} < {threshold}}}$ = {len(table_low)}", transform=ax.transAxes, fontsize=12, va='top')

    # Selecting the part of the table satisfying the condition on the threshold column and dropping the NaN values for the column to plot
    table_high = table[table[threshold_col] >= threshold].dropna(subset=[col])
    # Plotting the histogram for the part of the table selected
    table_high[col].hist(bins=bins[col], weights=compute_column_weights(table_high, col), ax=ax, edgecolor="red", label=rf"{threshold_col} $\geq$ {threshold}", histtype="step", linewidth=2)
    # Displaying the number of spectra used for the part of the table selected
    ax.text(0.80, 0.90,rf"$N_{{{threshold_col} \geq {threshold}}}$ = {len(table_high)}", transform=ax.transAxes, fontsize=12, va='top')

    # Adding the legend to the plot
    ax.legend(title=threshold_col, loc="upper left")

    # Returning to the main function
    return

# Function to plot the histogram of a given column for each category of spectra in the given table
def plot_sample_statistics_standard(table : pd.DataFrame, col : str, ax : plt.Axes) -> None:
    """
    This function plots the histogram of the given column.

    :param table: The table containing the spectra data to plot, with a column for the category
    of each spectrum and a column for the values to plot.
    :type table: pd.DataFrame
    :param col: The name of the column to plot the histogram for.
    :type col: str
    :param ax: The `matplotlib` axis to plot the histogram on.
    :type ax: plt.Axes
    :return: None
    """
    
    # Selecting the part of the table with non-NaN values for the column to plot
    table_to_plot = table.dropna(subset=[col])
    # Plotting the histogram
    table_to_plot[col].hist(bins=bins[col], weights=compute_column_weights(table_to_plot, col), ax=ax, edgecolor='black', histtype='step', linewidth=2)
    # Displaying the number of spectra used
    ax.text(0.90, 0.95,f"N = {len(table_to_plot)}", transform=ax.transAxes, fontsize=12, va='top')

    # Returning to the main function
    return
