"""
This module contains a class to represent a spectrum record for easier handling of the retrieved spectra data.
It allows for easier access and manipulation of the spectrum data and metadata.
"""

# Importing necessary libraries
from dataclasses import dataclass
import numpy as np

# Local imports
from src.desiqso.constants import H2_LYMAN_WERNER_BANDS, SNR_ESTIMATION_RANGE
from src.desiqso.utils.helpers import _is_valid_continuum

# Class representing a spectrum record for easier handling of the retrieved spectra data from the DESI-DR1 database
@dataclass
class SpectrumRecord:
    """
    Class representing a spectrum record for easier handling of the retrieved spectra data from the 
    DESI-DR1 database. It contains attributes representing the spectrum data (wavelength, flux, error) 
    and metadata (redshift, data release, model, spectype, specid, ra, dec). It also includes a property 
    to compute the inverse variance from the error array for easier handling of the uncertainty data and a 
    class method to create a `SpectrumRecord` instance from a FITS file data for easier conversion of 
    the loaded spectrum data into a python object to manipulate it for the spectra analysis.

    It contains the following attributes representing the spectrum data: 
    - wavelength: The wavelength array of the spectrum.
    - flux: The flux array of the spectrum.
    - err: The error array of the spectrum.
    - redshift: The redshift of the quasar in the spectrum.
    - redshift_err: The error on the redshift of the quasar in the spectrum.
    - redshift_wrn: The warning flag for the redshift of the quasar in the spectrum.
    - name: The name of the quasar in the spectrum (ex: 12:34:56.78+12:34:56.78).
    - data_release: The data release of the spectrum.
    - model: The model of the continuum provided.
    - mask: The mask of pixels to be masked in the spectrum.
    - spectype: The type of the spectrum (ex: QSO).
    - specid: The ID of the spectrum in the SPARCL database.
    - ra: The right ascension of the quasar in the spectrum.
    - dec: The declination of the quasar in the spectrum.
    - chi_2: The chi-squared value of the fit.
    - tsnr2_qso: The tSNR2 value of the quasar in the spectrum.

    This class also contains the following methods and properties:
    - ivar: A property to compute the inverse variance from the error array for easier handling of the uncertainty data.
    - continuum: A property to access rapidly the estimated constant continuum level.
    - filename: A property to access rapidly the file name of the spectrum (ex: desi_J123456.78+123456.78.fits).
    - from_fits: A class method to create a `SpectrumRecord` instance from a FITS file data.
    """

    # Class attributes representing the spectrum data and metadata
    wavelength:     np.ndarray
    flux:           np.ndarray
    err:            np.ndarray
    redshift:       float
    redshift_err:   float
    redshift_wrn:   float
    name:           str
    data_release:   str
    model:          np.ndarray
    mask:           np.ndarray
    spectype:       str
    specid:         float
    ra:             float
    dec:            float
    chi_2:          float
    tsnr2_qso:      float

    # Property to compute the inverse variance from the error array for easier handling of the uncertainty data
    @property
    def ivar(self) -> np.ndarray:
        """
        Compute the inverse variance from the error array for easier handling of the uncertainty data.
        The inverse variance is commonly used in astronomical data analysis to represent the uncertainty of measurements.

        :return: The inverse variance array computed from the error array.
        :rtype: np.ndarray
        """

        # Compute and return the inverse variance
        return 1 / (self.err ** 2)
    
    # Property to access rapidly the estimated constant continuum level
    @property
    def continuum(self) -> float:
        """
        Property to access rapidly the estimated constant continuum level.

        This property is computed on the fly by determining the Lyman-Werner region for 
        the spectrum using its redshift, and then computing the constant continuum level 
        as the 4/3 empirical correction of the 75th percentile of the flux values in the 
        Lyman-Werner region. This property is intended to be used for quick access to the 
        estimated constant continuum level without the need to compute it every time it is 
        needed.

        If the computed value is not valid (i.e. not finite or smaller than 0), an error is 
        raised, leading to return -1. for invalid values.

        :return float: The constant continuum level.
        """

        # Determining Lyman-Werner region for the spectrum using its redshift
        region = ((self.wavelength >= H2_LYMAN_WERNER_BANDS[0] * (1. + self.redshift)) & (self.wavelength <= H2_LYMAN_WERNER_BANDS[1] * (1. + self.redshift)))

        # Extract valid flux values (remove NaN/inf)
        flux_region = self.flux[region]
        flux_region = flux_region[np.isfinite(flux_region)]

        # Compute constant continuum value
        #continuum_value = np.quantile(flux_region, 0.75) * 4./3.
        #continuum_value = np.quantile(flux_region, 0.80)
        continuum_value = np.quantile(flux_region, 0.90)*1.1 - 1.3*np.median(self.err[region])

        # Check if the computed value is not valid, which is the case if it is finite and greater than 0
        if not _is_valid_continuum(continuum_value):
            # Return NaN for invalid values
            return np.nan

        # Return the constant continuum level
        return continuum_value

    # Property to access rapidly the estimated Signal-to-Noise Ratio (SNR) of the spectrum
    @property
    def snr(self) -> float:
        """
        This property is intended to be used for quick access to the estimated Signal-to-Noise Ratio (SNR)

        It computes the SNR outside the Lyman-Werner region for the spectrum using its redshift,
        and then computes the median of the flux divided by the error in the SNR estimation region.
        """
        
        # Determining another region for the spectrum to use for SNR estimation
        region_snr = (self.wavelength >= SNR_ESTIMATION_RANGE[0]*(1.+self.redshift)) & (self.wavelength <= SNR_ESTIMATION_RANGE[1]*(1.+self.redshift))
        # Compute (if possible) signal to noise ration (SNR) in the SNR estimation region using 
        # the flux and error arrays
        if self.flux[region_snr].size == 0 or self.err[region_snr].size == 0:
            # Return nan if the region is empty
            return np.nan
        # Else, compute SNR normally and proceed with the analysis
        else :
            return np.median(self.flux[region_snr]/self.err[region_snr].mean())

    # Property to access rapidly the estimated Continuum-to-Noise Ratio (CNR) of the spectrum
    @property
    def cnr(self) -> float:
        """"""
        # Determining Lyman-Werner region for the spectrum using its redshift
        region = ((self.wavelength >= H2_LYMAN_WERNER_BANDS[0] * (1. + self.redshift)) & (self.wavelength <= H2_LYMAN_WERNER_BANDS[1] * (1. + self.redshift)))
        # Compute Continuum-to-Noise Ratio (CNR) in the Lyman-Werner region using the estimated continuum level and the error array
        return self.continuum / np.median(self.err[region])
    
    # Property to access rapidly the file name of the spectrum
    @property
    def filename(self) -> str:
        """
        Property to access rapidly the file name of the spectrum. The file name
        is formatted as desi_JNAME.fits, where NAME is the name of the spectrum.

        :return str: The file name of the spectrum.
        """
        return f"desi_J{self.name.replace(':', '')}.fits"

    # Class method to create a `SpectrumRecord` instance from a FITS file data, for easier conversion of the loaded spectrum data into a python object to manipulate it for the spectra analysis
    @classmethod
    def from_fits(cls, hdul):
        """
        Create a `SpectrumRecord` instance from a FITS file data. This method is designed to facilitate 
        the conversion of the loaded spectrum data from local FITS files into a python object that can 
        be easily manipulated for the spectra analysis. It extracts the relevant data and metadata from 
        the FITS file and initializes a `SpectrumRecord` instance with this information.

        :param hdul: The HDUList object containing the FITS file data, which includes both the spectrum data and the header with metadata.
        :type hdul: `astropy.io.fits.HDUList`
        :return SpectrumRecord: A `SpectrumRecord` instance containing the spectrum data and metadata.
        """
        # Extract the relevant data and metadata from the FITS file
        data = hdul[1].data
        hdr = hdul[0].header
        # Return a `SpectrumRecord` instance initialized with the extracted data and metadata
        return cls(
            wavelength  =   data.field(0),
            flux        =   data.field(1) * 1e17,
            err         =   data.field(2) * 1e17,
            redshift    =   hdr.get("Z_EM", np.nan),
            redshift_err=   hdr.get("Z_ERR", np.nan),
            redshift_wrn=   hdr.get("Z_WARN", np.nan),
            name        =   hdr.get("NAME", "Unknown"),
            data_release=   hdr.get("DATAREL", "Unknown"),
            model       =   hdul["MODEL"].data,
            mask        =   hdul["MASK"].data,
            spectype    =   hdr.get("SPECTYPE", "QSO"),
            specid      =   hdr.get("SPECID", np.nan),
            ra          =   hdr.get("RA", np.nan),
            dec         =   hdr.get("DEC", np.nan),
            chi_2       =   hdr.get("CHI_2", np.nan),
            tsnr2_qso   =   hdr.get("TSNR2", np.nan),
        )    
