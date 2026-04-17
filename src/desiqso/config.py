"""
This module contains global configuration parameters for the program.
It contains parameters related to the following tasks:
- Matplotlib plots settings
- Spectra analysis (cross-correlation analysis, statistics analysis)
- Synthetic profiles generation (SNR estimation region, redshift range, etc.)
- Files paths (spectra data, synthetic profiles, etc.)
"""

# Matplotlib plots settings
settings = {
    'font.size'             : 16,
    'axes.linewidth'        : 2.0,
    'xtick.major.size'      : 12.0,
    'xtick.minor.size'      : 8.0,
    'xtick.major.width'     : 2.0,
    'xtick.minor.width'     : 1.5,
    'xtick.direction'       : 'in', 
    'xtick.minor.visible'   : True,
    'xtick.top'             : False,
    'ytick.major.size'      : 6.0,
    'ytick.minor.size'      : 4.0,
    'ytick.major.width'     : 2.0,
    'ytick.minor.width'     : 1.5,
    'ytick.direction'       : 'in', 
    'ytick.minor.visible'   : True,
    'ytick.right'           : True,
}

# ================
# Spectra analysis parameters
# ================

# Threshold for the p-value (log10) to consider a successful cross-correlation detection
PVALUE_THRESHOLD = -10.

# Threshold for the correlation parameter to consider a successful cross-correlation detection
CORRELATION_PARAM_THRESHOLD = 0.4

# Threshold for the Core Transmission to consider a successful cross-correlation detection
CORE_TRANSMISSION_THRESHOLD = 0.2

# ================
# Spectra retrieval parameters
# ================

# Minimum redshift for QSOs spectra retrieval
REDSHIFT_RANGE = [2.5, 10.]

# Number of QSOs spectra to retrieve by RA slice
NUMBER_OF_SPECTRA = 10000

# ================
# Cross-correlation parameters
# ================

# Parameter to know whether to use the basic preliminary synthetic profiles (from Pasquier) or not
USE_BASIC_SYNTHETIC_PROFILES = False

# SNR threshold for performing the cross-correlation analysis
SNR_THRESHOLD = 1.5

# Velocity range for the cross-correlation analysis (in km/s)
VELOCITY_RANGE = 2500.

# Number of redshift values to search for the H₂ absorption features in the cross-correlation analysis
NUM_REDSHIFT_VALUES = 100

# Threshold for the cross-correlation function to consider a H₂ absorption features in the synthetic profile
ABSORPTION_FEATURE_THRESHOLD = 0.95

# Threshold for the cross-correlation fonction to consider a core H₂ absorption feature in the synthetic profile
CORE_ABSORPTION_FEATURE_THRESHOLD = 0.2

# Flag to multiply the synthetic profiles by the estimated continuum level of the spectrum for the cross-correlation analysis
MULTIPLY_BY_CONTINUUM = True

# Flag to plot the correlation coefficients as a function of redshift for each spectrum
PLOT_CORRELATION_COEFFICIENTS = False

# Flag to plot the 2D distribution of the correlation coefficient and the core transmission for each redshift value during the cross-correlation analysis
PLOT_2D_DISTRIBUTION = False

# ================
# Files management
# ================

# Path to the folder containing the figures
FIGURES_FOLDER = "outputs/figures/"

# Path to the folder containing the results
RESULTS_FOLDER = "outputs/N0-20_J-0-1_sample-stat_all/"

# Path to the folder containing the spectra data
PRELIMINARY_DATA_PATH = "data/raw/preliminary/"

# Path to the spectra data folder
SPECTRA_DATA_FOLDER = "data/raw/spectra/"

# Path to the spectra plots folder
SPECTRA_PLOTS_FOLDER = RESULTS_FOLDER + "figures/spectra/"

# Path to the statistics plots folder
STATISTICS_PLOTS_FOLDER = RESULTS_FOLDER + "figures/statistics/"

# Path to the synthetic profiles folder
SYNTHETIC_PROFILES_FOLDER = "data/processed/synthetic_profiles/"

# Path to cross-correlation analysis results folder
CROSS_CORRELATION_RESULTS_FOLDER = RESULTS_FOLDER + "results/cross_correlation/"

# Path to cross-correlation analysis figures folder
CROSS_CORRELATION_FIGURES_FOLDER = RESULTS_FOLDER + "figures/cross-correlation/"

# Path to program evaluation results folder
DEPENDECY_RESULTS_FOLDER = RESULTS_FOLDER +"figures/dependency/"

# Path to the expected core transmissions file
EXPECTED_CORE_TRANSMISSIONS_PATH = "data/processed/dependency/expected_core_transmissions.txt"

# Path to the new candidates file
NEW_CANDIDATES_PATH = RESULTS_FOLDER + "results/new_candidates.txt"
