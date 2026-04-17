"""

"""

# Packages import
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Local imports
from src.desiqso.config import (settings, CORRELATION_PARAM_THRESHOLD,)
from src.desiqso.constants import (ColNames, Modes,)
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.models.profile import ProfileManager
from src.desiqso.utils.helpers import compute_column_weights

# Function to plot the initial sample statistics
def plot_sample_statistics() -> None:
    """"""

    # Loading cross-correlation analysis results
    AnalysisResults.load_results()
    # Loading all the synthetic profiles
    ProfileManager.load_all()

    #
    plt.rcParams.update(**settings)

    #
    bins = {
        ColNames.SNR: np.arange(0, 50, 0.25),
        ColNames.CONTINUUM: np.arange(0, 10, 0.5),
        ColNames.QSO_Z: np.arange(2.5, 6., 0.05),
        ColNames.Z: np.arange(2.5, 6., 0.05),
        ColNames.RA : np.arange(0, 360, 10),
        ColNames.DEC : np.arange(-40, 90, 5),
        ColNames.CORR_PARAM : np.arange(0, 1., 0.025),
        ColNames.CORE_TRANS : np.arange(-1., 1., 0.05),
        ColNames.GRADE : np.arange(0, 5, 1),
        ColNames.REL_SPEED : np.arange(-2500, 2500, 100),
    }

    # 
    low_snr_table = AnalysisResults._low_snr.copy()
    failed_table = AnalysisResults._failed.copy()
    #
    successful_table = AnalysisResults.results_survey(Modes.ALL, "best", {})
    valid_table = AnalysisResults.results_survey(Modes.VALID, "best", {})
    confirmed_table = AnalysisResults.results_survey(Modes.CONFIRMED, "best", {})
    rejected_table = AnalysisResults.results_survey(Modes.REJECTED, "best")


    #
    table_dict = {
#        "Low SNR"   : low_snr_table,
#        "Failed"    : failed_table,
#        "Successful": successful_table,
#        "Valid"     : valid_table,
#        "Confirmed" : confirmed_table,
#        "Rejected"  : rejected_table,
        "SNR Range" : successful_table
    }

    #
    cols_dict = {
        "Low SNR"   : [ColNames.SNR, ColNames.CONTINUUM, ColNames.QSO_Z, ColNames.RA, ColNames.DEC,],
        "Failed"    : [ColNames.SNR, ColNames.CONTINUUM, ColNames.QSO_Z, ColNames.RA, ColNames.DEC,],
        "Successful": [ColNames.SNR, ColNames.CONTINUUM, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Rejected"  : [ColNames.SNR, ColNames.CONTINUUM, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "Valid"     : [ColNames.SNR, ColNames.CONTINUUM, ColNames.QSO_Z, ColNames.Z, ColNames.RA, ColNames.DEC, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE, ColNames.REL_SPEED,],
        "SNR Range" : [ColNames.QSO_Z, ColNames.RA, ColNames.DEC, ColNames.CORR_PARAM, ColNames.CORE_TRANS, ColNames.GRADE,]
    }

    #
    for key, table in table_dict.items():

        #
        for col in cols_dict[key]:
            
            # Creating the figure and axis
            _, ax = plt.subplots(figsize=(12,8))

            # Plotting the histogram 
            if ColNames.CATEGORY in table.columns and key in ["Successful", "Valid"]:

                colors = {"confirmed":"green", "borderline":"orange", "rejected":"red", "other":"blue"}
                
                other_table = table[table[ColNames.CATEGORY] == "other"].dropna(subset=[col])
                bins = np.histogram_bin_edges(other_table[col], bins=bins[col])
                other_table[col].hist(bins=bins, weights=compute_column_weights(other_table, col), ax=ax, alpha=0.7, color="red", label="other", edgecolor='black')

                for category in table[ColNames.CATEGORY].unique():
                    category_table : pd.DataFrame = table[table[ColNames.CATEGORY] == category].dropna(subset=[col])
                    if category == "other" : 
                        continue
                    else :
                        category_table[col].hist(bins=bins, weights=compute_column_weights(category_table, col), ax=ax, alpha=0.5, label=category, color=colors[category], edgecolor='black')
                # 
                ax.legend(title=ColNames.CATEGORY, loc="upper left")
            
            #
            elif key == "SNR Range" :
                table_low_snr = table[table[ColNames.SNR] < 10].dropna(subset=[col])
                table_low_snr[col].hist(bins=bins[col], weights=compute_column_weights(table_low_snr, col), ax=ax, alpha=0.7, color='blue', edgecolor='black', label="SNR < 10")
                table_high_snr = table[table[ColNames.SNR] >= 10].dropna(subset=[col])
                table_high_snr[col].hist(bins=bins[col], weights=compute_column_weights(table_high_snr, col), ax=ax, alpha=0.7, color='orange', edgecolor='black', label="SNR >= 10")
                table_very_high_snr = table[table[ColNames.SNR] >= 15].dropna(subset=[col])
                table_very_high_snr[col].hist(bins=bins[col], weights=compute_column_weights(table_very_high_snr, col), ax=ax, alpha=0.7, color='red', edgecolor='black', label="SNR >= 15")
                ax.legend(title="SNR range", loc="upper left")
                # Displaying the number of spectra used
                ax.text(0.80, 0.95,rf"$N_{{SNR < 10}}$ = {len(table_low_snr)}", transform=ax.transAxes, fontsize=12, va='top')
                ax.text(0.80, 0.90,rf"$N_{{SNR >= 10}}$ = {len(table_high_snr)}", transform=ax.transAxes, fontsize=12, va='top')
                ax.text(0.80, 0.85,rf"$N_{{SNR >= 15}}$ = {len(table_very_high_snr)}", transform=ax.transAxes, fontsize=12, va='top')

            else:
                table_to_plot = table.dropna(subset=[col])
                table_to_plot[col].hist(bins=bins[col], weights=compute_column_weights(table_to_plot, col), ax=ax, alpha=0.7, color='blue', edgecolor='black')
                # Displaying the number of spectra used
                ax.text(0.80, 0.95,f"N = {len(table_to_plot)}", transform=ax.transAxes, fontsize=12, va='top')
            
            # Setting the labels and title of the plot
            plt.xlabel(col)
            plt.ylabel("Fraction of spectra")
            plt.title(f"Distribution of {col} for {key} spectra")
            # Setting the x-axis limits to better visualize the distribution
            match col:
                case ColNames.RA:
                    plt.xlim(0, 360)
                case ColNames.DEC:
                    plt.xlim(-40, 90)
                case ColNames.QSO_Z:
                    plt.xlim(left=2.45)
                case ColNames.Z:
                    plt.xlim(left=2.45)
                case ColNames.CORR_PARAM:
                    plt.axvline(x=CORRELATION_PARAM_THRESHOLD, color="red", linestyle="--", linewidth=2)
                case ColNames.REL_SPEED:
                    plt.xlim(-2500, 2500)
                case _:
                    pass
            # Adding a grid to the plot
            ax.grid(True, alpha=.5)
            # Show the plot and stop the execution until the plot is closed
            plt.show(block=True)
        