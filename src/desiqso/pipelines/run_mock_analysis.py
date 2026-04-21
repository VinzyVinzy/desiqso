""""""

# Local imports
from src.desiqso.analysis.mocks_spectra import (mock_analysis,)

# Entry point for the `run-mock-analysis` command
if __name__ == "__main__":

    # Calling the function to retrieve and save the list of spectra files with high SNR
    mock_spectra = mock_analysis()