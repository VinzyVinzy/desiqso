"""
This module contains functions to process and save spectra data retrieved from the DESI-DR1 database.

It includes functions to generate human-readable names for spectra based on their metadata, and 
to save the spectra data in local FITS files with updated headers containing relevant metadata 
for future reference.
"""

# Packages import
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
import astropy.units as units
import numpy as np
import os
from specutils import Spectrum
from tqdm import tqdm
from typing import Any

# Local imports
from src.desiqso.config import SPECTRA_DATA_FOLDER

# Ignore divide by zero warnings when computing uncertainties from inverse variance (which can be zero for some pixels)
np.seterr(divide = 'ignore')

# Function to process a single record of a retrieved spectra from the DESI-DR1 database
def process_spectrum_record(record: Any, count: int) -> None:
    """
    Function processing a single record of a retrieved spectrum from the DESI-DR1 database by 
    using `SparclClient()`. It attributes a human-readable name to the spectrum based on its 
    metadata and saves the spectrum data in a local file (`SPECTRA_DATA_FOLDER`).
    
    :param record: A record containing the metadata and data of a retrieved spectrum from the DESI-DR1 database.
    :type record: `sparcl.Results.Retrieved.Records`
    """
    # Attribute a human-readable to the spectra by calling the dedicated function `generate_spectrum_name`
    name = generate_spectrum_name(record.ra, record.dec)
    # Inform user of the processed spectrum
    tqdm.write(f"[INFO] Processing spectrum {count}: {name} (Redshift: z={record.redshift:.3f})")
    # Save the spectrum data in a local file using the `save_spectrum_data` function
    save_spectrum_data(record, name)

# Function to generate a human-readable name for a spectrum based on its metadata
def generate_spectrum_name(ra: float, dec: float) -> str:
    """
    Function generating a human-readable name for a spectrum based on its metadata, by 
    formatting the target coordinates (RA and DEC) into a string format commonly used 
    in astronomy (e.g., "Jhhmmss+ddmmss").

    :param ra: The right ascension of the target in degrees.
    :type ra: float
    :param dec: The declination of the target in degrees.
    :type dec: float
    :return: The human-readable name for the spectrum.
    :rtype: str
    """
    # Retrieve target coordinates with correct units
    ra = ra * units.degree
    dec = dec * units.degree
    # Create a `SkyCoord` object for the target to simplify coordinate formatting
    coord = SkyCoord(ra=ra, dec=dec)
    # Format the coordinates into a human-readable string using the `SkyCoord` formatting methods
    ra_str = coord.ra.to_string(unit=units.hour, sep=":", precision=2, pad=True, format="hms")
    dec_str = coord.dec.to_string(sep=":", precision=2, alwayssign=True, pad=True)
    # Renvoi du nom du spectre
    return ra_str + dec_str

# Function to save a spectrum data in a local file 
def save_spectrum_data(record: Any, name: str) -> None:
    """
    Save a spectrum data in a local file (`SPECTRA_DATA_FOLDER`) using the `Spectrum` class 
    from `specutils` package. The function also updates the `.fits file header with relevant 
    metadata for future reference.
    
    :param record: A record containing the metadata and data of a retrieved spectrum from the DESI-DR1 database.
    :type record: `sparcl.Results.Retrieved.Records`
    :param name: The human-readable name generated for the spectrum.
    :type name: str
    """
    
    # Ensure the existence of the output directory
    if not os.path.exists(SPECTRA_DATA_FOLDER):
        os.makedirs(SPECTRA_DATA_FOLDER)
    
    # Attribute file name and path to the spectrum data
    file_name = os.path.join(SPECTRA_DATA_FOLDER, f"desi_J{name.replace(':', '')}.fits")

    # Convert the spectrum data into a `Spectrum` object for easier saving
    spectrum = Spectrum(
        spectral_axis   = record.wavelength * units.AA,   # Wavelength data in Angstroms
        flux            = np.array(record.flux) * 1e-17 * units.erg / (units.s * units.cm**2 * units.AA), # Flux data in erg/s/cm^2/Angstrom
        uncertainty     = StdDevUncertainty(np.array(record.ivar)**(-0.5) * 1e-17 * units.erg / (units.s * units.cm**2 * units.AA)), # Uncertainty data derived from inverse variance, in erg/s/cm^2/Angstrom
        redshift        = record.redshift, # Redshift of the spectrum
    )

    # Save the spectrum data in a local FITS file using the `Spectrum` object's built-in saving method
    spectrum.write(file_name, overwrite=True)

    # Update FITS file header with relevant metadata for future reference
    with fits.open(file_name,mode='update') as hdul:

        # Save the continuum model given by the SPARCL database in another Image HDU
        hdul.append(fits.ImageHDU(data=record.model, name = "MODEL"))

        # Save the mask provided by the SPARCL database in another Image HDU
        hdul.append(fits.ImageHDU(data=record.mask, name = "MASK"))

        # Set the relevant metadata in the FITS file header
        hdul[0].header["Z_EM"]      = (record.redshift, "Redshift from SPARCL")
        hdul[0].header["Z_ERR"]     = (record.redshift_err, "Redshift uncertainty from SPARCL")
        hdul[0].header["Z_WARN"]    = (record.redshift_warning, "Redshift warning flag from SPARCL")
        hdul[0].header["NAME"]      = (name, "Human-readable name for the object")
        hdul[0].header["RA"]        = (record.ra, "Right Ascension in degrees")
        hdul[0].header["DEC"]       = (record.dec, "Declination in degrees")
        hdul[0].header["SPECID"]    = (record.specid, "Unique DESI identifier for the spectrum")
        hdul[0].header["SPECTYPE"]  = (record.spectype, "DESI spectral type")
        hdul[0].header["DATAREL"]   = (record.data_release, "DESI data release version")
        hdul[0].header["CHI_2"]     = (record.chi2, "Chi-squared value of the fit")
        hdul[0].header["TSNR2"]     = (record.tsnr2_qso, "TSNR2 value of the fit")
        hdul.flush()
