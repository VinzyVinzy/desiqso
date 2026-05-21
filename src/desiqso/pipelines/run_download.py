"""
This module contains the entry point for the `make download-spectra` command.
"""
import os
# Local imports
from src.desiqso.data.spectra_downloader import (download_preliminary_spectra, retrieve_spectra_from_database, check_spectra_downloaded)
from src.desiqso.data.magnitudes import download_magnitudes

# Entry point for the `make download-spectra` command
if __name__ == "__main__":

    # Calling the function to check that all the quasar spectra are correctly downloaded
    ra_list = check_spectra_downloaded()

    # Calling the function to download first the preliminary analysis results
    download_preliminary_spectra()

    # Calling the function to retrieve all quasar spectra with a redshift above 2.5 from the DESI-DR1 database
    retrieve_spectra_from_database(ra_list)

    # Calling the function to download the magnitudes data for the quasar spectra from the DESI-DR1 photometric database using TAP queries
    download_magnitudes()
