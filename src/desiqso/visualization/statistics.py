"""
This module contains functions to plot statistics of the cross-correlation analysis, in order to make accurate diagnostics of the program.
It currently contains the following functions:
- plot_correlation_vs_redshift: plots the correlation coefficients as a function of redshift for a given spectrum, through the cross-correlation analysis.
- plot_corrcoeff_vs_coretrans_2d: plots the 2D distribution of the correlation coefficient and core transmissions for a given spectrum at multiple redshift through the cross-correlation analysis.
- plot_distribution: plots the distribution of diverse parameters for a given list of processed spectra.
- select_spectra_for_statistics: selects a subset of spectra for statistics analysis in the `plot_distribution` function.
- plot_statistics: manages the behavior of the `plot-statistics` command, organizing the execution of the `plot_distribution` function.
"""

# Importing necessary libraries
from astropy.convolution import (convolve, Box1DKernel,)
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import pandas as pd
from scipy.stats import (binned_statistic, gaussian_kde,)
from tqdm import tqdm

# Local imports
from src.desiqso.config import (settings, CROSS_CORRELATION_FIGURES_FOLDER, CORRELATION_PARAM_THRESHOLD, STATISTICS_PLOTS_FOLDER,)
from src.desiqso.constants import (C_KMS, Categories, ColNames, COLUMN_FILE_LABELS, Modes,)
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.models.profile import (Profile, ProfileManager,)
from src.desiqso.utils.helpers import normalize


# ================================
# Global variables for the plots
# ================================

# Global variable to store the sizes of the plots
plot_sizes = {
    ColNames.CORR_PROB  : [0, 0], 
    ColNames.CORR_COEFF : [-0.02, 1.02], 
    ColNames.CORR_PARAM : [-0.02, 1.02], 
    ColNames.CORE_TRANS : [-0.55, 0.75],
    ColNames.Z          : [2.5, 6.0], 
    ColNames.QSO_Z      : [2.5, 6.0], 
    ColNames.SNR        : [0, 50], 
    ColNames.GRADE      : [-0.5, 6.5], 
    ColNames.REL_SPEED  : [-2600, 2600],
}
# Defining the alpha value for each group
alphas = {
    Categories.CONFIRMED : 1.0,
    Categories.BORDERLINE: .90,
    Categories.REJECTED  : .75,
    Categories.OTHER     : .50,
}

# Defining the markers for each group
markers = {
    Categories.CONFIRMED : "o",
    Categories.BORDERLINE: "v",
    Categories.REJECTED  : "X",
    Categories.OTHER     : "s"
}
# Defining the markers size for each group
sizes = {
    Categories.CONFIRMED : 80,
    Categories.BORDERLINE: 50,
    Categories.REJECTED  : 50,
    Categories.OTHER     : 20
}
# Defining the edgecolors for each group
edgecolors = {
    Categories.CONFIRMED : "red",
    Categories.BORDERLINE: "none",
    Categories.REJECTED  : "none",
    Categories.OTHER     : "none"
}



# Function to plot the correlation coefficients as a function of redshift for a given spectrum
def plot_correlation_vs_redshift(z_values : np.ndarray, correlation_coefficients : np.ndarray, core_transmissions : np.ndarray, core_transmission_levels : list, filename : str, profile : Profile, redshift : float, delta_z : float, SNR : float, show : bool = False, save : bool = True) -> None:
    """
    Plot the correlation coefficients as a function of redshift for a given spectrum.

    :param z_values: The redshift values of the spectrum.
    :type z_values: `np.ndarray`
    :param correlation_coefficients: The correlation coefficients between the spectrum and the synthetic H2 profile.
    :type correlation_coefficients: `np.ndarray`
    :param core_transmissions: The core transmissions of the synthetic H2 profile.
    :type core_transmissions: `np.ndarray`
    :param core_transmission_levels: The levels of the core transmissions.
    :type core_transmission_levels: `list`
    :param filename: The filename of the spectrum.
    :type filename: `str`
    :param profile_name: The name of the synthetic H2 profile used for the analysis.
    :type profile_name: `str`
    :param redshift: The redshift of the spectrum.
    :type redshift: `float`
    :param delta_z: The delta redshift of the spectrum.
    :type delta_z: `float`
    :param SNR: The signal-to-noise ratio of the spectrum.
    :type SNR: `float`

    :return: None

    Notes
    -----
    This function defaults to NOT showing the plot and saving it as a PNG file 
    in the `results/plots/` folder. The filename of the saved plot is derived from 
    the input `filename` parameter.
    """
    
    # =========
    # Configuration parameters
    # =========

    # Load the preliminary analysis results to retrieve the redshift from the preliminary analysis, if available
    AnalysisResults.load_preliminary_results(verbose=False)

    # Updating matplotlib parameters
    settings["xtick.top"]   = False
    settings["ytick.right"] = False
    plt.rcParams.update(**settings)

    # Extracting the name of the spectrum from the filename
    name = filename[5:-5]

    # =========
    # Computing and plotting
    # =========

    # Computing the values to plot
    correlation_coefficients = convolve(correlation_coefficients, Box1DKernel(3))
    core_transmissions       = convolve(core_transmissions, Box1DKernel(3))
    diff                     = normalize(correlation_coefficients-core_transmissions)
    prod                     = normalize(correlation_coefficients*(1-core_transmissions))
    core_transmission_levels = np.array(core_transmission_levels)

    # Initialize figure
    fig, ax1 = plt.subplots(figsize=(14,14))
    # Plotting the correlation coefficients as a function of redshift
    ax1.plot(z_values, correlation_coefficients, color="black", alpha=0.7, linewidth=2, label="A: Correlation coefficient")
    # Plotting the core transmission as a function of redshift
    ax2 = ax1.twinx()
    ax2.plot(z_values, core_transmissions, color="blue", alpha=0.7, linewidth=2, label="B: Core transmission")
    ax2.plot(z_values, diff, color="orange", alpha=0.7, linewidth=2, label="C: Normalized difference (A-B)")
    ax2.plot(z_values, prod, color="green", alpha=0.7, linewidth=2, label="D: Normalized product(A*(1-B))")
    for i in range(len(core_transmission_levels[0])):
        ax2.plot(z_values, convolve(core_transmission_levels[:,i], Box1DKernel(3)), alpha=0.3, linestyle="dashed", label=f"Core transmission for level J={i}")
    
    # Plotting the redshift values kept as best redshift
    max_correlation_index = np.nanargmax(correlation_coefficients)
    #ax2.axvline(x=z_values[max_correlation_index],                                                          color="cyan",    alpha=0.5, label="A: Corr. coeff. max.")
    #ax2.axvline(x=z_values[np.nanargmin(core_transmissions)],                                               color="green",  alpha=0.5, label="B: Core trans. min.")
    #ax2.axvline(x=z_values[np.nanargmax(-core_transmissions+correlation_coefficients)],                     color="orange", alpha=0.5, label="C: Maximum difference")
    #ax2.axvline(x=np.mean([z_values[max_correlation_index], z_values[np.nanargmin(core_transmissions)]]),   color="purple", alpha=0.5, label="D: Mean of A and B")
    
    # =========
    # Formatting plot
    # =========

    # Plotting the redshift value from the preliminary analysis, if available
    if filename in AnalysisResults._preliminary_results["confirmed_candidates"]["File Name"].tolist():
        # Retrieving the best fit redshift value from the preliminary analysis
        row = AnalysisResults._preliminary_results["confirmed_candidates"][AnalysisResults._preliminary_results["confirmed_candidates"]["File Name"] == filename].iloc[0]
        # Add a vertical line at the redshift value from the preliminary analysis
        ax1.axvline(x=row["Best fit redshift"], color="red", linewidth=5, linestyle="dashed", label="Redshift from preliminary analysis")
    
    # Displaying the delta-z value on the plot, along with the redshift value from the preliminary analysis, the best fit redshift value and the SNR level of the spectrum
    ax1.text(0.02, 0.98, rf"$\Delta z = {delta_z:.6f}$"+"\n"+rf"$z_{{QSO}} = {redshift:.6f}$"+"\n"+rf"$z_{{best}} = {z_values[max_correlation_index]:.6f}$"+"\n"+rf"$\mathrm{{SNR}} = {SNR:.1f}$", transform=ax1.transAxes, fontsize=12, va="top")

    # Defining the conversion functions
    def z_to_v(z):
        return C_KMS * ((z - redshift) / (1 + redshift))
    def v_to_z(v):
        return (v * (1 + redshift) / C_KMS) + redshift
    # Adding a secondary x-axis for velocity
    secax = ax1.secondary_xaxis("top", functions=(z_to_v, v_to_z))
    secax.set_xlabel(r"Velocity (km/s)")

    # Setting the y-axis limits
    ax1.set_ylim(-1.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    # Labelling the plot
    ax1.set_xlabel("Redshift")
    ax1.set_ylabel("Correlation coefficient")
    ax2.set_ylabel("Excess core transmission\nNormalized quantities")

    # Setting the title
    title = f"Correlation coefficient / Excess core transmission as a function of redshift for spectrum\n{name}"
    title = title + f"\n({profile.legend_label})\n\n"
    plt.title(title)
    # Adding a grid
    ax1.grid(True)

    # Combining the legends of both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # =========
    # Saving and showing the plot
    # =========
    
    # Saving the plot, if required
    if save:
        plot_filename = os.path.join(CROSS_CORRELATION_FIGURES_FOLDER, f"redshift_vs_correlation_values/{name}_{profile.name}_snr-{SNR:.3f}.png")
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename)
    # Showing the plot, if required
    if show:
        plt.show()

    # Closing the plot to free memory
    plt.close(fig)

    # Return to the main programm
    return

# Function to plot the 2D distribution of the correlation coefficient and core transmissions for a given spectrum at multiple redshift through the cross-correlation analysis
def plot_corrcoeff_vs_coretrans_2d(z_values : np.ndarray, correlation_coefficients : np.ndarray, core_transmissions : np.ndarray, filename : str, profile : Profile, redshift : float, delta_z : float, SNR : float, show : bool = False, save : bool = True) -> None:
    """
    This function will plot the 2D distribution of the correlation coefficient and core transmissions for a given 
    spectrum at multiple redshift through the cross-correlation analysis, with a color scale corresponding to the 
    redshift values. It aims to provide the user with a visual representation of the correlation between the 
    correlation coefficient and core transmissions for different redshift values in order to help them in 
    identifying the best fit redshift.

    :param z_values: Array of redshift values corresponding to the correlation coefficient and core transmissions.
    :type z_values: np.ndarray
    :param correlation_coefficients: Array of correlation coefficients corresponding to the redshift values.
    :type correlation_coefficients: np.ndarray
    :param core_transmissions: Array of core transmissions corresponding to the redshift values.
    :type core_transmissions: np.ndarray
    :param filename: File name of the spectrum.
    :type filename: str
    :param profile: `Profile` object corresponding to the spectrum.
    :type profile: Profile
    :param redshift: Redshift value of the spectrum.
    :type redshift: float
    :param delta_z: Delta-z value of the spectrum.
    :type delta_z: float
    :param SNR: Signal-to-noise ratio of the spectrum.
    :type SNR: float
    :param show: Flag indicating whether to show the plot.
    :type show: bool
    :param save: Flag indicating whether to save the plot.
    :type save: bool
    """

    # =========
    # Configuration parameters
    # =========

    # Load the preliminary analysis results to retrieve the redshift from the preliminary analysis, if available
    AnalysisResults.load_preliminary_results(verbose=False)

    # Updating matplotlib parameters
    settings["xtick.top"]   = True
    settings["ytick.right"] = True
    plt.rcParams.update(**settings)

    # Extracting the name of the spectrum from the filename
    name = filename[5:-5]

    # =========
    # Plotting
    # =========

    # Initialize figure
    fig, ax = plt.subplots(figsize=(14,14))
    # Plotting the 2D distribution of the correlation coefficient and core transmissions for a given spectrum at multiple redshift through the cross-correlation analysis, with a color scale corresponding to the redshift values
    sc = ax.scatter(correlation_coefficients, core_transmissions, c=z_values, cmap="viridis", s=75)
    # Displaying colorbar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Redshift (z)")

    # =========
    # Formatting plot
    # =========

    # Plotting the redshift value from the preliminary analysis, if available
    if filename in AnalysisResults._preliminary_results["confirmed_candidates"]["File Name"].tolist():
        # Retrieve the redshift value from the preliminary analysis
        z = AnalysisResults._preliminary_results["confirmed_candidates"][AnalysisResults._preliminary_results["confirmed_candidates"]["File Name"] == filename].iloc[0]["Best fit redshift"]
        # Retrieve the index corresponding to the point whose redshift is closer to the redshift from the preliminary analysis
        idx = np.abs(z_values - z).argmin()
        # Plotting the redshift value from the preliminary analysis on the 2D distribution
        ax.scatter(correlation_coefficients[idx], core_transmissions[idx], color="red", marker="x", s=500, label="Redshift from preliminary analysis")
        # Displaying legend
        ax.legend(loc="upper right")

    # Retrieving the index of the point with the maximum correlation coefficient
    max_correlation_index = np.nanargmax(correlation_coefficients)

    # Displaying the delta-z value on the plot, along with the redshift value from the preliminary analysis, the best fit redshift value and the SNR level of the spectrum
    ax.text(0.02, 0.98, rf"$\Delta z = {delta_z:.6f}$"+"\n"+rf"$z_{{QSO}} = {redshift:.6f}$"+"\n"+rf"$z_{{best}} = {z_values[max_correlation_index]:.6f}$"+"\n"+rf"$\mathrm{{SNR}} = {SNR:.1f}$", transform=ax.transAxes, fontsize=12, va="top")

    # Setting x-axis and y-axis limits
    ax.set_xlim(correlation_coefficients.min() - 0.05, correlation_coefficients.max() + 0.05)
    ax.set_ylim(core_transmissions.min() - 0.05, core_transmissions.max() + 0.05)

    # Setting x-axis and y-axis labels
    ax.set_xlabel("Correlation coefficient")
    ax.set_ylabel("Excess core transmission")

    # Setting the title
    title = f"Correlation coefficient as a function of excess core transmission\n{name}"
    title = title + f"\n({profile.legend_label})\n\n"
    plt.title(title)
    # Adding a grid
    ax.grid(True, alpha=0.3)

    # =========
    # Saving and showing the plot
    # =========

    # Saving the plot, if required
    if save:
        plot_filename = os.path.join(CROSS_CORRELATION_FIGURES_FOLDER, f"correlation-coefficient_vs_core-transmission/{name}_{profile.name}_SNR-{SNR:.2f}.png")
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename)
    # Showing the plot, if required
    if show:
        plt.show()

    # Closing the plot to free memory
    plt.close(fig)

    # Return to main program
    return

# Function to plot statistics distribution as scatter (with contour levels) or bin plot using a list of tuples containing the names of the statistics to plot
def plot_distribution(plot_pairs : list[tuple[str,str]], thresholds : dict[str, tuple[float]], profile_name : str, mode : str = "all",) -> None:
    """
    Main function to plot statistics distribution as scatter (with contour levels) 
    or bin plot using a list of tuples containing the names of the statistics to 
    plot. It also applies the given thresholds to the data as a mask before plotting.

    :param plot_pairs: List of tuples containing the names of the statistics to plot.
    :type plot_pairs: list[tuple[str,str]]
    :param thresholds: Dictionary containing the minimum and maximum values for each statistic.
    :type thresholds: dict[str, tuple[float]]
    :param profile_name: Name of the synthetic profile.
    :type profile_name: str
    :param result_table: Table containing the results of the cross-correlation analysis.
    :type result_table: pd.DataFrame
    :param mode: Mode of the analysis. Can be "all" or "random".
    :type mode: str
    """

    # =========
    # Configuration parameters
    # =========

    # Turn off interactive mode to prevent visual artifacts
    plt.ioff()

    # Load data from the results table using the survey tool
    data = AnalysisResults.results_survey(mode, thresholds_dict=thresholds, profile_name=profile_name,)

    # Loop on the data and the associated mode
    for x_key, y_key in tqdm(plot_pairs, desc="Plotting statistics", unit="plot"):

        # Inform user
        tqdm.write(f"[INFO] Plotting {x_key} vs {y_key}...\n")

        # Setting plot type depending on the x-axis values
        PLOT_TYPE = "bin" if x_key in [ColNames.Z, ColNames.SNR, ColNames.GRADE] else "scatter"
    
        # Creating the figure and axis
        _, ax = plt.subplots(figsize=(12,8))

        # Retrieving the x and y data from the filtered data
        x_data = data[x_key]
        y_data = data[y_key]

        # ==========
        # Contours
        # ==========

        # Plotting the contours only in "scatter" mode
        if PLOT_TYPE == "scatter":
            # Calling the function plotting the contours
            plot_distribution_with_contours(x_data, y_data, ax=ax)
        
        # If the plot mode is set to "bin"
        if PLOT_TYPE == "bin":
            # If the x-key is "grade"
            if x_key == "grade":
                # Calling the function dealing with the "grade" case
                plot_grade_bin(x_data, y_data, ax=ax)

            # Else, for the regular cases
            else:
                # Calling the dedicated function
                plot_standard_bin(x_data, y_data, ax=ax)
        # If the plot mode is set to "scatter"
        else:
            # Calling the function to plot the scatter
            plot_scatter(data, x_key, y_key, ax=ax)
        
        # =========
        # Plot formatting
        # =========

        # Plotting the line corresponding to the threshold value of the correlation parameter
        if x_key == ColNames.CORR_PARAM:
            plt.axvline(x=CORRELATION_PARAM_THRESHOLD, color="black", linestyle="--", linewidth=2)
        if y_key == ColNames.CORR_PARAM:
            plt.axhline(y=CORRELATION_PARAM_THRESHOLD, color="black", linestyle="--", linewidth=2)

        # Displaying the number of spectra used
        ax.text(0.05, 0.05,f"N = {len(data[x_key])}",transform=ax.transAxes,fontsize=12,va='top')
        
        # Generating plot title
        title = f"{x_key} vs. {y_key} distribution"
        # Adding the result table name to the title
        title += f"\nSynthetic profile used: {profile_name}"
        # Adding the mode to the title (if not all)
        if mode != "all":
            title += f"\nMode: {mode}"
        # Adding the threshold values to the title
        for key, (min_val, max_val) in thresholds.items():
            if min_val is not None or max_val is not None:
                title += f"\n{key} threshold: {min_val if min_val is not None else 'None'} - {max_val if max_val is not None else 'None'}"

        # Plot title and axis legend
        plt.title(title)
        plt.xlabel(x_key)
        plt.ylabel(y_key)
        # Adding a grid to the plot
        plt.grid(True, alpha=.5)
        # Plotting the legend only in "scatter" mode
        if PLOT_TYPE == "scatter":
            # Defining the marker legend
            marker_legend = [Line2D([0], [0], marker=marker, color="black", linestyle="None", markersize=8, label=f"{group}", markeredgecolor=edgecolors[group], linewidth=1.,) for group, marker in markers.items()]
            # Adding the legend to the plot
            plt.legend(handles=marker_legend, title="Groups", loc="upper right")

        # Compute the plot limits using percentiles to be less affected by outliers
        x_low, x_high = np.percentile(x_data, [0.01, 99.99])
        y_low, y_high = np.percentile(y_data, [0.01, 99.99])
        x_min, x_max = min(plot_sizes[x_key][0], x_low), max(plot_sizes[x_key][1], x_high)
        y_min, y_max = min(plot_sizes[y_key][0], y_low), max(plot_sizes[y_key][1], y_high)

        # Set the plot x-axis limits dynamically with mode
        if x_key == ColNames.CORR_COEFF:
            plt.xlim(x_min-0.03, x_max+0.03)
        elif x_key == ColNames.CORR_PROB:
            plt.xlim(x_min-0.7, x_max+0.7)
        elif x_key in [ColNames.Z, ColNames.QSO_Z]:
            plt.xlim(2.4, x_max+0.1)
        elif x_key == ColNames.SNR:
            plt.xlim(x_min-1, x_max+1)
        else:
            plt.xlim(x_min-0.05, x_max+0.05)

        # Set the plot y-axis limits dynamically with mode, only for "scatter" plotting mode
        if PLOT_TYPE == "scatter":
            if y_key == ColNames.CORR_COEFF:
                plt.ylim(y_min-0.05, y_max+0.05)
            elif y_key == ColNames.CORR_PROB:
                plt.ylim(y_min-0.7, y_max+0.7)
            elif y_key in [ColNames.QSO_Z, ColNames.Z]:
                plt.ylim(2.4, y_max+0.1)
            elif y_key == ColNames.SNR:
                plt.ylim(y_min-1, y_max+1)
            else:
                plt.ylim(y_min-0.05, y_max+0.05)

        # ============
        # Saving plot
        # ============
            
        # Generating plot name
        plot_name = f"{COLUMN_FILE_LABELS[x_key]}_vs_{COLUMN_FILE_LABELS[y_key]}_distribution"
        # Adding thresholds to the plot name
        for key, (min_val, max_val) in thresholds.items():
            if min_val is not None or max_val is not None:
                plot_name += f"_{COLUMN_FILE_LABELS[key]}-{min_val if min_val is not None else 'None'}-{max_val if max_val is not None else 'None'}"
        # Generating plot path, adding the mode and profile name to the path
        savepath = os.path.join((STATISTICS_PLOTS_FOLDER), f"{profile_name}_{mode}/")
        # If the directory does not exist, create it
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(savepath + f"{plot_name}.png", dpi=600)

        # Inform the user
        tqdm.write(f"[INFO] Plot saved to {savepath}{plot_name}\n")

# Function to plot a 2D contours of the distribution in a scatter plot
def plot_distribution_with_contours(x_data : pd.DataFrame, y_data : pd.DataFrame, ax : plt.Axes) -> None:
    """
    This function plots on the provided axis the selected 2D distribution contours using a 2D Gaussian Kernel Density Estimate.

    :param x_data: The x-axis data to plot.
    :type x_data: pd.DataFrame
    :param y_data: The y-axis data to plot.
    :type y_data: pd.DataFrame
    :param ax: The axis to plot on.
    :type ax: plt.Axes
    """

    # Computing the minimum and maximum values of the data distribution
    xmin, xmax = min(x_data), max(x_data)
    ymin, ymax = min(y_data), max(y_data)
    
    # If there are no valid spectra
    if len(x_data) == 0:
        # Inform user
        print("[WARNING] No valid spectra found with the provided parameters.\n")
        # Exit program with an error
        os._exit(1)

    # Creating the kernel density estimate using a 2D Gaussian Kernel
    values = np.vstack([x_data, y_data])
    kde = gaussian_kde(values)

    # Creating the grid fro the contour levels
    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    # Evaluating the density on the grid and reshaping the resulting kernel density estimate
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions), X.shape)    

    # Sorting the resulting kernel density estimate
    Z_sorted = np.sort(Z.ravel())[::-1]
    # Obtaining the cumulative distribution
    cdf = np.cumsum(Z_sorted)
    # Normalizing the cumulative distribution
    cdf /= cdf[-1]

    # Initializing the contour levels
    levels = []
    # Loop on levels corresponding to 1σ, 2σ, 3σ
    for sigma in [0.393, 0.865, 0.989]:
        # Obtaining the level corresponding to the sigma
        levels.append(Z_sorted[np.searchsorted(cdf, sigma)])

    # Sorting the levels
    levels = np.sort(levels)

    # Plotting the contours and adding labels
    contours = ax.contour(X, Y, Z, levels=levels, colors=["r","g","b"], linewidths=2.5)
    fmt = {levels[0]: r"3$\sigma$", levels[1]: r"2$\sigma$", levels[2]: r"1$\sigma$"}
    ax.clabel(contours, fmt=fmt, inline=True, fontsize=10)

# This function plots a binned scatter plot for each grade
def plot_grade_bin(x_data : pd.DataFrame, y_data : pd.DataFrame, ax : plt.Axes) -> None:
    """
    This function plots a binned scatter plot for each grade (x-axis values).

    :param x_data: The x-axis data to plot (the grades).
    :type x_data: pd.DataFrame
    :param y_data: The y-axis data to plot.
    :type y_data: pd.DataFrame
    :param ax: The axis to plot on.
    :type ax: plt.Axes
    """

    # Initializing the lists containing the means and the standard deviations for each grade
    means = []
    stds = []
    # Loop on the grades
    for grade in set(x_data):
        y_g = [y for x, y in zip(x_data, y_data) if x == grade]
        means.append(np.mean(y_g))
        stds.append(np.std(y_g))
    # Plotting the error bar plot
    ax.errorbar(sorted(set(x_data)), means, yerr=stds, fmt="o", color="black", ecolor="black", capsize=4, capthick=1)

# This function plots a binned scatter plot for the provided data
def plot_standard_bin(x_data : pd.DataFrame, y_data : pd.DataFrame, ax : plt.Axes) -> None:
    """
    This function plots a binned scatter plot for the provided data. The x-axis values 
    are binned in 20 bins and the y-axis values are plotted as error bars.

    :param x_data: The x-axis data to plot.
    :type x_data: pd.DataFrame
    :param y_data: The y-axis data to plot.
    :type y_data: pd.DataFrame
    :param ax: The axis to plot on.
    :type ax: plt.Axes
    """

    # Creating the binned statistic
    bin_means, bin_edges, _ = binned_statistic(x_data, y_data, statistic="mean", bins=20)
    bin_std, _, _ = binned_statistic(x_data, y_data, statistic="std", bins=bin_edges)
    # Computing the bin centers and widths
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = (bin_edges[1:] - bin_edges[:-1]) / 2
    # Plotting the error bar plot
    ax.errorbar(bin_centers, bin_means, xerr=bin_widths, yerr=bin_std, fmt="o", color="black", ecolor="black", capsize=4, capthick=1)

# This function plots a scatter plot for the provided data
def plot_scatter(data : pd.DataFrame, x_key : str, y_key : str, ax : plt.Axes) -> None:
    """
    Plots a scatter plot representing the provided data distribution.

    :param data: The data to plot.
    :type data: pd.DataFrame
    :param x_key: The key corresponding to the x-axis values.
    :type x_key: str
    :param y_key: The key corresponding to the y-axis values.
    :type y_key: str
    :param ax: The axis to plot on.
    :type ax: plt.Axes
    """

    categories = [Categories.OTHER, Categories.REJECTED, Categories.BORDERLINE, Categories.CONFIRMED]

    # Loop on the data groups
    for category in categories:
        # Filtering the data for the current group
        group = data[data[ColNames.CATEGORY] == category]
        # Retrieving the x and y values related to the current data group
        x_values = group[x_key]
        y_values = group[y_key]
        grades   = group[ColNames.GRADE]
        # Scatter plot of the current data group
        plt.scatter(x_values, y_values, c=grades, cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=6), marker=markers[category], s=sizes[category], alpha=alphas[category], edgecolors=edgecolors[category], linewidths=1.,)
    
    # Creating the colorbar and initializing it
    sm = cm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(vmin=0, vmax=6))
    sm.set_array([])
    # Adding the colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax)
    # Labeling the colorbar
    cbar.set_label("Grade")
    # Setting the colorbar ticks
    cbar.set_ticks(range(7))

# Entry point for `plot-statistics` command
def plot_statistics(plot_pairs : list[tuple[str, str]], thresholds : dict[str, tuple[float]], mode : str = Modes.ALL) -> None:
    """
    This function manages the behavior of the `plot-statistics` command. It contains
    an option to plot the evolution of certain distribution with the SNR. It takes as 
    input a list of tuples containing the names of the statistics to plot.

    :param plot_pairs: List of tuples containing the names of the statistics to plot.
    :type plot_pairs: list[tuple[str, str]]
    """
    
    # Loading cross-correlation analysis results
    AnalysisResults.load_results()
    # Loading all the synthetic profiles
    ProfileManager.load_all()

    # Loop on the result tables
    for profile_name in ProfileManager._profiles.keys():

        # Calling function to plot the p-value and core transmission distribution of all the processed spectra
        plot_distribution(
            plot_pairs      =   plot_pairs,    # Pairs of analysis parameters to plot
            thresholds      =   thresholds,
            profile_name    =   profile_name,
            mode            =   mode,
        )
