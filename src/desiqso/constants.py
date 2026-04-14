"""
This module contains constants used in the code.
"""

from enum import StrEnum

# Wavelengths of the H₂ Lyman and Werner bands in the rest frame (in Angstroms)
H2_LYMAN_WERNER_BANDS = [1035.0, 1130.0]

# Wavelength range for SNR estimation (in Angstroms)
SNR_ESTIMATION_RANGE = [1400.0, 1500.0]

# DESI mean resolution power over the wavelength range of interest
DESI_RESOLUTION_POWER = 2650    # Base formula: np.mean(np.array([2000., 3300.]))

# Light speed (in km/s)
C_KMS =  299792.458   # Base formula, from astropy.constants module: c.to("km/s").value

# Enumeration of column names to prevent typos
class ColNames(StrEnum):
    """
    This class contains the column names used in the code.
    """

    FILENAME        =   "File name"
    NAME            =   "Name"
    RA              =   "RA (deg)"
    DEC             =   "DEC (deg)"
    QSO_Z           =   "QSO Redshift"
    Z               =   "Redshift"
    CORR_COEFF      =   "Correlation coefficient"
    CORR_PROB       =   "Correlation probability"
    CORE_TRANS      =   "Excess core transmission"
    J_CORE_TRANS    =   "J core transmissions"
    CORR_PARAM      =   "Correlation parameter"
    SNR             =   "SNR"
    CONTINUUM       =   "Continuum"
    BEST_FIT_FLAG   =   "Best fit flag"
    DETAILS         =   "Details"
    GRADE           =   "Grade"
    REL_SPEED       =   "Relative speed"
    CATEGORY        =   "Category"
    IS_VALID        =   "Valid flag"
    PROFILE         =   "Profile name"

# Enumeration of analysis modes to prevent typos
class Modes(StrEnum):
    """
    This class contains the analysis modes used in the code.
    """

    ALL         = "all"
    RANDOM      = "random"
    VALID       = "valid"
    NEW         = "new"
    PRELIMINARY = "preliminary"
    CONFIRMED   = "confirmed"
    REJECTED    = "rejected"
    BORDERLINE  = "borderline"

# List of analysis modes associated with preliminary analysis
PREL_LIST = {Modes.CONFIRMED, Modes.BORDERLINE, Modes.REJECTED}
