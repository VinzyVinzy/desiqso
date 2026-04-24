"""
This module contains the entry point for the `run-mock-analysis` command.
"""

# Local imports
from src.desiqso.analysis.mocks_spectra import (mock_analysis, mock_spectra_statistics_plotting,)

# Entry point for the `run-mock-analysis` command
if __name__ == "__main__":

    # Calling the function to retrieve and save the list of spectra files with high SNR
    mock_spectra = mock_analysis(profile_ntot=19., use_same_profile=False)

    # Calling the function to plot the statistics of the mock spectra used for the analysis
    mock_spectra_statistics_plotting(mock_spectra = mock_spectra, profile_ntot=19.)