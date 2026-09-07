"""
This module contains the entry point for the `make download-spectra` command.

It first checks the status of the downloaded spectra and then proceeds to download the necessary data.
Then, it downloads the magnitudes and physical data for the quasar spectra from the databases associated 
with the DESI-DR1 program, using TAP queries.
"""

# Local imports
from src.desiqso.data.magnitudes import download_magnitudes
from src.desiqso.data.physical_data import download_physical_data
from src.desiqso.data.spectra_downloader import (download_preliminary_spectra, retrieve_spectra_from_database, check_spectra_downloaded)

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

    # Calling the function to download the physical data for the quasar spectra from the DESI-DR1 database using TAP queries
    download_physical_data()
