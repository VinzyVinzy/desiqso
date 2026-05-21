"""
This module contains constants used in the code.
"""

# Packages import
from enum import StrEnum

# Wavelengths of the H₂ Lyman and Werner bands in the rest frame (in Angstroms)
H2_LYMAN_WERNER_BANDS = [1035.0, 1130.0]
#H2_LYMAN_WERNER_BANDS = [1060.0, 1130.0]   # Use with NUMBER_OF_BANDS = 4 to perform the analysis with only 4 bands

# Number of bands to use in the cross-correlation analysis
NUMBER_OF_BANDS : int = 6

# Wavelength range for SNR estimation (in Angstroms)
SNR_ESTIMATION_RANGE = [1400.0, 1500.0]

# DESI mean resolution power over the wavelength range of interest
DESI_RESOLUTION_POWER = 2650    # Base formula: np.mean(np.array([2000., 3300.]))

# Light speed (in km/s)
C_KMS =  299792.458   # Base formula, from astropy.constants module: c.to("km/s").value

# Enumeration for categories names to prevent typos
class Categories(StrEnum):
    """
    This class contains the categories names used in the code.
    """

    CONFIRMED = "confirmed"
    UNSURE    = "unsure"
    REJECTED  = "rejected"
    OTHER     = "other"

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
    CNR             =   "CNR"
    BEST_FIT_FLAG   =   "Best fit flag"
    DETAILS         =   "Details"
    GRADE           =   "Grade"
    REL_SPEED       =   "Relative speed"
    CATEGORY        =   "Category"
    IS_VALID        =   "Valid flag"
    PROFILE         =   "Profile name"

# Dictionary of the columns labels for the filenames of the plots to save
COLUMN_FILE_LABELS = {
    ColNames.RA              : "ra",
    ColNames.DEC             : "dec",
    ColNames.QSO_Z           : "qso-z",
    ColNames.Z               : "abs-z",
    ColNames.CORR_COEFF      : "corr-coeff",
    ColNames.CORR_PROB       : "corr-prob",
    ColNames.CORE_TRANS      : "core-trans",
    ColNames.CORR_PARAM      : "corr-param",
    ColNames.SNR             : "snr",
    ColNames.CONTINUUM       : "continuum",
    ColNames.CNR             : "cnr",
    ColNames.GRADE           : "grade",
    ColNames.REL_SPEED       : "rel-speed",
}

# Enumeration of column names for the magnitudes table to prevent typos
class MagColNames(StrEnum):
    """
    This class contains the column names for the magnitudes table used in the code.
    """

    NAME        =   "Name"
    TARGETID    =   "Target ID"
    RA          =   "RA (deg)"
    DEC         =   "DEC (deg)"
    G_FLUX      =   "Legacy Survey g-band flux"
    R_FLUX      =   "Legacy Survey r-band flux"
    Z_FLUX      =   "Legacy Survey z-band flux"
    MW_TRANS_G  =   "Legacy Survey g-band Milky Way transmission"
    MW_TRANS_R  =   "Legacy Survey r-band Milky Way transmission"
    MW_TRANS_Z  =   "Legacy Survey z-band Milky Way transmission"
    W1_FLUX     =   "WISE W1 band flux"
    W2_FLUX     =   "WISE W2 band flux"
    W3_FLUX     =   "WISE W3 band flux"
    W4_FLUX     =   "WISE W4 band flux"
    MW_TRANS_W1 =   "WISE W1 band Milky Way transmission"
    MW_TRANS_W2 =   "WISE W2 band Milky Way transmission"
    MW_TRANS_W3 =   "WISE W3 band Milky Way transmission"
    MW_TRANS_W4 =   "WISE W4 band Milky Way transmission"
    GAIA_G_MAG  =   "Gaia g-band magnitude"
    GAIA_BP_MAG =   "Gaia bp-band magnitude"
    GAIA_RP_MAG =   "Gaia rp-band magnitude"

# Enumeration of analysis modes to prevent typos
class Modes(StrEnum):
    """
    This class contains the analysis modes used in the code.
    """

    ALL         = "all"
    RANDOM      = "random"
    VALID       = "valid"
    NEW         = "new"
    VISUAL      = "visual"
    CONFIRMED   = Categories.CONFIRMED
    REJECTED    = Categories.REJECTED
    UNSURE      = Categories.UNSURE
    OTHER       = Categories.OTHER
    SAMPLE      = "sample"
    CANDIDATES  = "candidates"

# List of analysis modes associated with visual inspection
VISUAL_LIST = {Modes.CONFIRMED, Modes.UNSURE, Modes.REJECTED}
