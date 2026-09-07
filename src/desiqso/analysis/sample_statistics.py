"""
This module contains the functions to plot the initial sample statistics for each category of spectra, 
using the associated colors for each category (confirmed, unsure, rejected and other), and displaying 
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
from src.desiqso.constants import (Categories, ColNames, COLUMN_FILE_LABELS, Modes,)
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.models.profile import ProfileManager
from src.desiqso.utils.helpers import compute_column_weights

# ================================
# Global variables for plots configuration
# ================================

# Defining the colors for each category
colors = {
    Categories.CONFIRMED:    "green", 
    Categories.UNSURE   :    "orange", 
    Categories.REJECTED :    "red", 
    Categories.OTHER    :    "blue",
}

# Defining the bins for the histograms of each column
bins = {
    ColNames.SNR        : np.arange(0, 50.25, 0.25),
    ColNames.CNR        : np.arange(0, 10.5, 0.25),
    ColNames.QSO_Z      : np.arange(2.5, 6.05, 0.05),
    ColNames.Z          : np.arange(2.5, 6.05, 0.05),
    ColNames.RA         : np.arange(0, 370, 10),
    ColNames.DEC        : np.arange(-40, 95, 5),
    ColNames.CORR_PARAM : np.arange(0., 1.00, 0.02),
    ColNames.CORR_COEFF : np.arange(-0.05, 1.025, 0.025),
    ColNames.CORE_TRANS : np.arange(-1., 1., 0.05),
    ColNames.GRADE      : np.arange(-0.5, 7.5, 1),
    ColNames.REL_SPEED  : np.arange(-2600, 2600, 200),
}

# Definig the y positions for the text displaying the number of spectra for each category in the histograms
y_pos_dict = {
    Categories.OTHER     :    0.95, 
    Categories.CONFIRMED :    0.90, 
    Categories.UNSURE    :    0.85, 
    Categories.REJECTED  :    0.80,
}

# Function to plot the initial sample statistics
def plot_sample_statistics() -> None:
    """
    This function plots the distribution of diverse parameters for different samples of spectra. When
    relevent, it also colors the plots according to the category of the spectra.
    """

    # ================================
    # Configuration and data loading
    # ================================

    # Loading cross-correlation analysis results
    AnalysisResults.load_results()
    # Loading all the synthetic profiles
    ProfileManager.load_all()

    # Updating the matplotlib settings
    settings["xtick.top"] = True
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
    other_table      = AnalysisResults.results_survey(Modes.OTHER, "best", {})
    new_table        = AnalysisResults.results_survey(Modes.NEW, "best", {})
    candidates_table = AnalysisResults.results_survey(Modes.CANDIDATES, "best", {})
    parent_table     = AnalysisResults.results_survey(Modes.ALL, "best", {ColNames.SNR : (3.0, None), ColNames.QSO_Z : (2.6, None)})
    visual_table     = AnalysisResults.results_survey(Modes.VISUAL, "best", {})


    # Defining the dictionary of the tables to plot for each category of spectra
    table_dict = {
        "Low SNR"                    : low_snr_table,       # Plotting the spectra with SNR so low the analysis was not performed
        "Failed"                     : failed_table,        # Plotting the spectra that failed the analysis for various reasons
        "Successful"                 : successful_table,    # Plotting the spectra that passed the analysis with color coding for the categories
        "Valid"                      : valid_table,         # Plotting the valid spectra (that passed the analysis AND the CORR_PARAM threshold)
        "Confirmed"                  : confirmed_table,     # Plotting the confirmed candidates
        "Rejected"                   : rejected_table,      # Plotting the rejected candidates
        "Other"                      : other_table,         # Plotting the other candidates
        "SNR Range"                  : successful_table,    # Plotting the spectra that passed the analysis for various SNR thresholds
        "New Candidates"             : new_table,           # Plotting the valide spectra that were not visually inspected
        "All"                        : successful_table,    # Plotting all the spectra without any color coding for the categories
        "Candidates"                 : candidates_table,    # Plotting the spectra that were visually inspected and classified as confirmed or unsure
        "Visual"                     : visual_table,        # Plotting the spectra that were visually inspected
        "Candidates VS Parent Sample": parent_table,        # Plotting all the spectra and color coding for the other vs H2 candidates spectra
        "Candidates VS Rejected"     : visual_table,        # Plotting all the spectra and color coding for the rejected candidates vs H2 candidates spectra
        "Visual Categories"          : visual_table,        # Plotting all the spectra that were visually inspected and color coding for the visual inspection categories
        "Detected VS Parent Sample"  : parent_table,        # Plotting all the spectra and color coding for the detected vs H2 candidates spectra and color coding for the visual inspection categories
    }

    # Defining the dictionary of the columns to plot for each category of spectra
    cols_dict = {
        "Low SNR"                    : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z,             ColNames.RA, ColNames.DEC,],
        "Failed"                     : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z,             ColNames.RA, ColNames.DEC,],
        "Successful"                 : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Confirmed"                  : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Other"                      : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Rejected"                   : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Valid"                      : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "SNR Range"                  : [ColNames.SNR,               ColNames.QSO_Z,             ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE,],
        "New Candidates"             : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "All"                        : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Candidates"                 : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Visual"                     : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Candidates VS Parent Sample": [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Candidates VS Rejected"     : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Visual Categories"          : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Detected VS Parent Sample"  : [ColNames.SNR, ColNames.CNR, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_COEFF, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
    }

    # ================================
    # Plotting the histograms for each category of spectra and column
    # ================================

    # Dictionnary of the tables selected for the plots
    selected_dict = {"Detected VS Parent Sample" : table_dict["Detected VS Parent Sample"]}

    # Looping over the categories of spectra and columns to plot and plotting the histogram for each category and column
    for key, table in tqdm(selected_dict.items(), desc="Plotting sample statistics", unit="category"):

        # Looping over the columns to plot for the category of spectra
        for col in cols_dict[key]:
            
            # Creating the figure and axis
            _, ax = plt.subplots(figsize=(12,8))

            # Matching the category of spectra to plot with the associated plotting function
            match key:
                
                # For the successful and valid spectra, plotting the histogram with the associated categories colors
                case "Successful" | "Valid" | "Visual Categories":
                    # Plotting the histogram with the associated categories colors
                    plot_sample_statistics_categories(table, col, ax)

                # For the successful spectra, plotting the histogram with the associated threshold for the SNR to measure its effect
                case "SNR Range":
                    # Plotting the histogram with the associated function
                    plot_sample_statistics_threshold(table, col, 3., ColNames.SNR, ax)
                
                # For the H2 candidates, plotting their distributions with regard to the rest of the spectra
                case "Candidates VS Parent Sample":
                    # Plotting the histogram with the associated function
                    plot_sample_statistics_candidates_vs_parent(table, col, ax)

                # For the H2 candidates, plotting their distributions with regard to the rejected candidates spectra
                case "Candidates VS Rejected":
                    # Plotting the histogram with the associated function
                    plot_sample_statistics_candidates_vs_rejected(table, col, ax)

                # For the detected candidates, plotting their distribution with regard to the rest of the parent sample
                case "Detected VS Parent Sample":
                    # Plotting the histogram with the associated function
                    plot_sample_statistics_detected_vs_parent(table, col, ax)

                # For all the other categories
                case _:
                    # Plotting the histogram with the associated function
                    plot_sample_statistics_standard(table, col, ax)
            
            # Setting the labels and title of the plot
            ax.set_xlabel(col)
            if key not in ["Candidates VS Parent Sample", "Detected VS Parent Sample"]:
                plt.ylabel("Percentage of total spectra")
            plt.title(f"Distribution of {col} for {key} spectra")

            # Setting the x-axis limits to better visualize the distribution
            plt.xlim(bins[col][0], bins[col][-1])

            # Setting the y-axis to display the percentage instead of the fraction
            if key not in ["Candidates VS Parent Sample", "Detected VS Parent Sample"]:
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

            # Adding a vertical line to indicate the threshold for the correlation parameter if the column is the correlation parameter
            if col == ColNames.CORR_PARAM:
                ax.axvline(x=CORRELATION_PARAM_THRESHOLD, color="k", linestyle="--", linewidth=3)
                        
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
def plot_sample_statistics_candidates_vs_parent(table: pd.DataFrame, col: str, ax: plt.Axes) -> None:
    """
    This function plots the histogram of the given column for each category of spectra in the given table,
    using the associated colors for each category and displaying the number of spectra used for each category 
    in the plot. The "candidate" category is defined as the combination of "confirmed" and "unsure" categories
    while the "parent sample" category is defined as the combination of "other" and "rejected" categories 
    respecting the selection criteria for visual inspection (SNR > 3.0 and z_qso > 2.6).

    :param table: The table containing the spectra data to plot, with a column for the category
    of each spectrum and a column for the values to plot.
    :type table: pd.DataFrame
    :param col: The name of the column to plot the histogram for.
    :type col: str
    :param ax: The `matplotlib` axis to plot the histogram on.
    :type ax: plt.Axes
    :returns None: This function does not return any value.
    """

    # Creating a second y-axis to plot the histogram of the parent sample with the count of spectra in each bin
    ax2 = ax.twinx() 

    # Selecting the part of the table with non-NaN values for the "negative" category and column to plot
    other_table = table[table[ColNames.CATEGORY].isin([Categories.OTHER, Categories.REJECTED, Categories.UNSURE, Categories.CONFIRMED])].dropna(subset=[col])
    # Plotting the histogram for the "negative" category and column with the associated color and label, using the weights to plot the fraction of spectra in each bin instead of the count
    other_table[col].hist(bins=bins[col], ax=ax, label="parent sample", edgecolor=colors[Categories.OTHER], histtype='step', linewidth=2)
    # Displaying the number of spectra used for the "negative" category and column
    ax.text(0.85, 0.95, rf"N$_{{parent}}$ = {len(other_table)}", transform=ax.transAxes, fontsize=12, va='top')

    # Selecting the part of the table with non-NaN values for the "candidate" category and column to plot
    candidates_table = table[table[ColNames.CATEGORY].isin([Categories.CONFIRMED, Categories.UNSURE])].dropna(subset=[col])
    # Plotting the histogram for the "candidate" category and column with the associated color and label, using the weights to plot the fraction of spectra in each bin instead of the count
    candidates_table[col].hist(bins=bins[col], ax=ax2, label="candidates", edgecolor="green", histtype='step', linewidth=2)
    # Displaying the number of spectra used for the "candidate" category and column
    ax2.text(0.85, 0.90, rf"N$_{{candidates}}$ = {len(candidates_table)}", transform=ax2.transAxes, fontsize=12, va='top')

    # Merge legends from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles2 + handles1, labels2 + labels1, title=ColNames.CATEGORY, loc="upper left")

    # Setting the y-axis labels for both axes
    ax.set_ylabel("Number of spectra (Parent sample)")
    ax2.set_ylabel("Number of spectra (Candidates)")

    # Turning off the grid for the second y-axis to avoid overloading the plot
    ax2.grid(False)

    # Returning to the main function
    return

# Function to plot the histogram of a given column for each category of spectra in the given table
def plot_sample_statistics_candidates_vs_rejected(table: pd.DataFrame, col: str, ax: plt.Axes) -> None:
    """
    This function plots the histogram of the given column for each category of spectra in the given table,
    using the associated colors for each category and displaying the number of spectra used for each category 
    in the plot. The "candidate" category is defined as the combination of "confirmed" and "unsure" categories.

    :param table: The table containing the spectra data to plot, with a column for the category
    of each spectrum and a column for the values to plot.
    :type table: pd.DataFrame
    :param col: The name of the column to plot the histogram for.
    :type col: str
    :param ax: The `matplotlib` axis to plot the histogram on.
    :type ax: plt.Axes
    :returns None: This function does not return any value.
    """

    # Selecting the part of the table with non-NaN values for the "negative" category and column to plot
    rejected_table = table[table[ColNames.CATEGORY] == Categories.REJECTED].dropna(subset=[col])
    # Plotting the histogram for the "negative" category and column with the associated color and label, using the weights to plot the fraction of spectra in each bin instead of the count
    rejected_table[col].hist(bins=bins[col], weights=compute_column_weights(rejected_table, col), ax=ax, label=Categories.REJECTED, edgecolor=colors[Categories.REJECTED], histtype='step', linewidth=2)
    # Displaying the number of spectra used for the "negative" category and column
    ax.text(0.85, 0.90, rf"N$_{{rejected}}$ = {len(rejected_table)}", transform=ax.transAxes, fontsize=12, va='top')

    # Selecting the part of the table with non-NaN values for the "candidate" category and column to plot
    candidates_table = table[table[ColNames.CATEGORY].isin([Categories.CONFIRMED, Categories.UNSURE])].dropna(subset=[col])
    # Plotting the histogram for the "candidate" category and column with the associated color and label, using the weights to plot the fraction of spectra in each bin instead of the count
    candidates_table[col].hist(bins=bins[col], weights=compute_column_weights(candidates_table, col), ax=ax, label="candidates", edgecolor="green", histtype='step', linewidth=2)
    # Displaying the number of spectra used for the "candidate" category and column
    ax.text(0.85, 0.85, rf"N$_{{candidates}}$ = {len(candidates_table)}", transform=ax.transAxes, fontsize=12, va='top')

    # Adding the legend to the plot
    ax.legend(title=ColNames.CATEGORY, loc="upper left")

    # Returning to the main function
    return

# Function to plot the histogram of a given column for each category of spectra in the given table
def plot_sample_statistics_detected_vs_parent(table: pd.DataFrame, col: str, ax: plt.Axes) -> None:
    """
    This function plots the  cumulative histogram of the given column for each category of spectra in the 
    given table, using the associated colors for each category and displaying the number of spectra used 
    for each category in the plot. Only the spectra respecting the selection criteria for visual 
    inspection (SNR > 3.0 and z_qso > 2.6) are plotted here.

    :param table: The table containing the spectra data to plot, with a column for the category
    of each spectrum and a column for the values to plot.
    :type table: pd.DataFrame
    :param col: The name of the column to plot the histogram for.
    :type col: str
    :param ax: The `matplotlib` axis to plot the histogram on.
    :type ax: plt.Axes
    :returns None: This function does not return any value.
    """

    # Defining the categories list
    categories = [Categories.OTHER, Categories.CONFIRMED, Categories.UNSURE, Categories.REJECTED]

    # Extract data for each category
    data = []
    labels = []
    color_list = []

    # Looping over the categories and plotting the histogram for each category with the associated color
    for category in categories:
        # Selecting the category and column to plot, dropping the NaN values
        category_table : pd.DataFrame = table[table[ColNames.CATEGORY] == category].dropna(subset=[col])
        # Appending the data, labels, and colors to the lists
        data.append(category_table[col])
        labels.append(f"{category} (N = {len(category_table)})")
        color_list.append(colors[category])
    
    # Plotting the stacked histogram for the "candidate" category and column with the associated color and label, using the weights to plot the fraction of spectra in each bin instead of the count
    ax.hist(data, bins=bins[col], stacked=True, histtype="stepfilled", edgecolor="black", facecolor=color_list, alpha=0.7, linewidth=1.2, label=labels)
    # Displaying the number of spectra used for the "parent sample" category and column
    ax.text(0.85, y_pos_dict[Categories.OTHER], rf"N$_{{parent}}$ = {len(table)}", transform=ax.transAxes, fontsize=12, va='top')
    # Displaying the number of spectra used for the "confirmed" category and column
    ax.text(0.85, y_pos_dict[Categories.CONFIRMED], rf"N$_{{confirmed}}$ = {len(table[table[ColNames.CATEGORY] == Categories.CONFIRMED])}", transform=ax.transAxes, fontsize=12, va='top')
    # Displaying the number of spectra used for the "rejected" category and column
    ax.text(0.85, y_pos_dict[Categories.REJECTED], rf"N$_{{rejected}}$ = {len(table[table[ColNames.CATEGORY] == Categories.REJECTED])}", transform=ax.transAxes, fontsize=12, va='top')
    # Displaying the number of spectra used for the "unsure" category and column
    ax.text(0.85, y_pos_dict[Categories.UNSURE], rf"N$_{{unsure}}$ = {len(table[table[ColNames.CATEGORY] == Categories.UNSURE])}", transform=ax.transAxes, fontsize=12, va='top')

    # Adding the legend
    ax.legend(title=ColNames.CATEGORY, loc="upper left")

    # Setting the y-axis labels for both axes
    ax.set_ylabel("Stacked number of spectra")
    
    # Setting the y-axis scales to log for both axes to better visualize the distribution
    ax.set_yscale("log")

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
