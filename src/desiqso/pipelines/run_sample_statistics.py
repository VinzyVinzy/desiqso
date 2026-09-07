"""
This module contains the entry point for the `make sample-statistics` command.

It allows to plot on the flight the distribution of diverse parameters for different samples of spectra.
"""

# Local imports
from src.desiqso.analysis.sample_statistics import plot_sample_statistics

# Entry point for the `make sample-statistics` command
if __name__ == "__main__":

    # Calling the function to plot the statistics of the sample of spectra used for the analysis
    plot_sample_statistics()
