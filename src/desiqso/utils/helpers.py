"""
This module contains utility functions used in the project.

Currently, it contains the following functions:
- `normalize(x: np.ndarray)`: normalizes an array using the min-max normalization.
- `get_profile_characteristics(filename: str)`: retrieves the characteristics of a synthetic profile from its filename.
- `parse_cell(x)`: parses a cell from a `pd.DataFrame` object.
- `compute_grade(J_core_transmissions: list[float])`: computes the grade of a synthetic profile using its core transmissions.
- `_is_valid_continuum(continuum: float)`: checks if a given value is a valid continuum value.
- `compute_relative_speed(z_abs: float, z_qso: float)`: computes the relative speed between two redshifts.
- `_is_valid(row: pd.Series)`: checks if a row in a `pd.DataFrame` object is associated with a valid spectrum.
- `compute_column_weights(table: pd.DataFrame, column: str)`: computes the weights for a given column in a table.
- `compute_excitation_temperature(T_exc0: float)`: computes the excitation temperature for each rotational level from the ground level.
- `compute_legacy_magnitude(flux: pd.Series, mw_transmission: pd.Series)`: computes the magnitude from a flux.
- `compute_constant_continuum(wavelength: np.ndarray, flux: np.ndarray, err: np.ndarray, redshift: float)`: computes a constant continuum level from a spectrum.
"""

# Importing necessary libraries
import ast
import numpy as np
import pandas as pd

# Local imports
from src.desiqso.config import (CORE_TRANSMISSION_THRESHOLD, SNR_THRESHOLD, CORRELATION_PARAM_THRESHOLD,)
from src.desiqso.constants import (H2_LYMAN_WERNER_BANDS, C_KMS, ColNames,)

# Utility function to normalize an array using the min-max normalization
def normalize(x : np.ndarray) -> np.ndarray:
    """
    Normalize an array by subtracting the minimum and dividing by the range.

    :param x: The array to normalize.
    :type x: np.ndarray
    :return np.ndarray: The normalized array.
    """

    # Compute and return the normalized array
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# Utility function to obtain all caracteritics of a synthetic profile from its filename
def get_profile_characteristics(filename : str) -> tuple[str] : 
    """
    Retrieves the characteristics of a synthetic profile from its filename.

    The filename is expected to follow the naming convention :
    "h2_profile_res-V_ntot-W_J-0-1..._Texc-X_b-Y_pix-Z.npy"

    :param filename: The filename of the synthetic profile.
    :return tuple[str]: A tuple containing the characteristics of the synthetic profile.
    """
    
    # If the file name is one of the basic profiles
    if filename == "synth_a":
        return "?", 20.0, "?"
    elif filename == "synth_b":
        return "?", 21.0, "?"
    # If the filename is one of the synthetic profiles
    else :  
        # Split the filename
        name_split = filename.split("_")
        # Retrieve the characteristics
        T_exc = name_split[5][5:]
        J = name_split[4][2:]
        Ntot = name_split[3][5:]
        res = name_split[2][4:]
        b_param = name_split[6][2:]
        pix_size = name_split[7][4:-4]
        # Return the characteristics
        return T_exc, J, Ntot, res, b_param, pix_size

# Utility function to parse a cell from a `pd.DataFrame`
def parse_cell(x):
    """
    This utility function is used to parse a cell from a `pd.DataFrame` object. 
    It takes a cell as input and returns a list of values. It is currently used 
    to parse the `J_core_transmissions` column of the `AnalysisResults` DataFrame.
    """

    # Check the type of the cell
    if isinstance(x, list):
        # If the cell is a list, return the list
        return [float(v) for v in x]
    # If the cell is a string
    if isinstance(x, str):
        # Remove the numpy wrapper
        x = x.replace("np.float64", "")
        # Parse the string as a list
        return [float(v) for v in ast.literal_eval(x)]
    
    # Return the parsed cell
    return x

# Utility function to compute the grade of a synthetic profile using its core transmissions
def compute_grade(J_core_transmissions : list[float]) -> int:
    """
    Utility function to compute the grade of a synthetic profile using its core transmissions.

    :param J_core_transmissions: The core transmissions of the synthetic profile.
    :type J_core_transmissions: list[float]
    :return int: The grade of the synthetic profile (between 0 and 5).
    """

    # Return the number of core transmissions below the threshold
    return sum(core_trans < CORE_TRANSMISSION_THRESHOLD for core_trans in J_core_transmissions)

# Utility function to check that the computed continuum is valid
def _is_valid_continuum(continuum : float) -> bool:
    """
    Utility function to check that the computed continuum is valid.
    A continuum is valid if it is finite and greater than 0.

    :param continuum: The computed continuum.
    :type continuum: float
    :return bool: True if the computed continuum is valid, False otherwise.
    """

    # Return True if the computed continuum is valid, False otherwise
    return np.isfinite(continuum) and continuum > 0

# Utility function to compute the relative speed between two redshifts
def compute_relative_speed(z_abs : float, z_qso : float) -> float:
    """
    Utility function to compute the relative speed between two redshifts.

    :param z1: The first redshift.
    :type z1: float
    :param z2: The second redshift.
    :type z2: float
    :return float: The relative speed between the two redshifts.
    """

    # Return the relative speed between the two redshifts
    return C_KMS * ((z_abs - z_qso) / (1 + z_qso))

# Utility function to check if a row is associated with a valid spectrum
def _is_valid(row : pd.Series) -> int:
    """
    This function returns 1 if the row is associated with a valid spectrum, 0 otherwise.

    :param row: The row to check.
    :type row: pd.Series
    :return int: 1 if the row is associated with a valid spectrum, 0 otherwise.
    """

    # Condition a row needs to satisfy to be considered valid
    condition = (row[ColNames.SNR] > SNR_THRESHOLD) & (row[ColNames.CORR_PARAM] > CORRELATION_PARAM_THRESHOLD) & (row[ColNames.BEST_FIT_FLAG] == 1)
   
    # Return 1 if the row is valid, 0 otherwise
    return 1 if condition else 0

# Utility function to compute the weights for a given column in a table
def compute_column_weights(table : pd.DataFrame, column : str) -> np.ndarray:
    """
    This function computes the weights for a given column in a table. The weights are computed as the inverse of the number of occurrences of each value in the column.

    :param table: The table to compute the weights for.
    :type table: pd.DataFrame
    :param column: The column to compute the weights for.
    :type column: str
    :return np.ndarray: The weights for each value in the column.
    """

    # Return the computed weights
    return np.ones_like(table[column]) / len(table)

# Utility function to compute the excitation temperature for each rotational level from the ground level
def compute_excitation_temperature(T_exc0 : float) -> list[float]:
    """
    This function computes the excitation temperature for each rotational level from the ground level.

    :param T_exc: The excitation temperature of the ground level.
    :type T_exc: float
    :return list[float]: The list of excitation temperatures for all the rotational levels.
    """

    # List of excitation temperatures
    Texc = [T_exc0, T_exc0, 240., 240., 240., 240., 240.]

    # Return the excitation temperature for the rotational level
    return Texc

# Utility function to compute the magnitude from a flux
def compute_legacy_magnitude(flux : pd.Series, mw_transmission : pd.Series) -> pd.Series:
    """
    This function computes the magnitude from a flux using the formula : m = -2.5 * log10(flux/mw_transmission) + 22.5

    :param flux: The flux to compute the magnitude from.
    :type flux: pd.Series
    :param mw_transmission: The Milky Way transmission in the band.
    :type mw_transmission: pd.Series
    :return pd.Series: The computed magnitudes.
    """

    # Return the computed magnitudes
    return -2.5 * np.log10(flux/mw_transmission) + 22.5

# Utility function to compute a constant continuum level from a spectrum
def compute_constant_continuum(wavelength : np.ndarray, flux : np.ndarray, err : np.ndarray, redshift : float) -> float:
    """
    This function computes a constant continuum level from a spectrum.

    :param wavelength: The wavelength array of the spectrum.
    :type wavelength: np.ndarray
    :param flux: The flux array of the spectrum.
    :type flux: np.ndarray
    :param err: The error array of the spectrum.
    :type err: np.ndarray
    :param redshift: The redshift of the spectrum.
    :type redshift: float
    :return float: The computed constant continuum level.
    """

    # Determining Lyman-Werner region for the spectrum using its redshift
    region = ((wavelength >= H2_LYMAN_WERNER_BANDS[0] * (1. + redshift)) & (wavelength <= H2_LYMAN_WERNER_BANDS[1] * (1. + redshift)))

    # Extract valid flux values (remove NaN/inf)
    flux_region = flux[region]
    flux_region = flux_region[np.isfinite(flux_region)]

    # Compute constant continuum value
    #continuum_value = np.quantile(flux_region, 0.75) * 4./3.
    #continuum_value = np.quantile(flux_region, 0.80)
    continuum_value = np.nanquantile(flux[region], 0.90)*1.1 - 1.3*np.nanmedian(err[region])

    # Check if the computed value is not valid, which is the case if it is finite and greater than 0
    if not _is_valid_continuum(continuum_value):
        # Return NaN for invalid values
        return np.nan
    
    return continuum_value
