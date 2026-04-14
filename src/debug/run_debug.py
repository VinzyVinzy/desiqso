"""
This module contains the entry point of the debug script. It currently calls the function to 
check that all the quasar spectra are correctly downloaded.
"""

# Local imports
from src.debug.spectra_downloader_debug import check_spectra_downloaded

# Entry point of the script
if __name__ == "__main__":

    # Calling the function to check that all the quasar spectra are correctly downloaded
    check_spectra_downloaded()