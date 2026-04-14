"""
This module contains functions to load spectra data from local `.fits` files as `SpectrumRecord` instances.
It can currently be done using its filename
"""

# Importing necessary libraries
from astropy.io import fits
import os

# Local imports
from src.desiqso.config import SPECTRA_DATA_FOLDER
from src.desiqso.models.spectrum import SpectrumRecord
    
# Function to load a spectrum as a `SpectrumRecord` instance from its filename
def load_spectrum_from_filename(filename : str) -> SpectrumRecord:
    """
    Load a spectrum as a `SpectrumRecord` instance from its filename.

    :param filename: The filename of the spectrum to load.
    :type filename: `str`

    :return: A `SpectrumRecord` instance representing the loaded spectrum.
    :rtype: `SpectrumRecord`
    """

    # Load the spectrum data from the current file using `astropy.io.fits`
    with fits.open(os.path.join(SPECTRA_DATA_FOLDER, filename), memmap=False) as hdul:
        # Convert the loaded spectrum data into a python object to easily manipulate it for the 
        # spectra analysis and returns it
        return SpectrumRecord.from_fits(hdul)
