"""
This module contains functions to compute masks for H₂ absorption features in the synthetic profiles 
for the spectra analysis. The first computed mask corresponds to the presence of absorption features
whereas the second mask corresponds only to the core of strong absorption features. The function also
rebin the synthetic profile onto the wavelength grid of the observed spectrum using interpolation.
"""

# Importing necessary libraries
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# Local imports
from src.desiqso.config import ABSORPTION_FEATURE_THRESHOLD, CORE_ABSORPTION_FEATURE_THRESHOLD
from src.desiqso.constants import H2_LYMAN_WERNER_BANDS
from src.desiqso.models.profile import Profile

# Function to compute masks for H₂ absorption features and rebin synthetic profile
def compute_h2_absorption_masks(wavelength : np.ndarray, redshift : float, profile : Profile, mask : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute masks for H₂ absorption features from a synthetic profile.

    This function loads a synthetic H₂ absorption profile (if not already
    loaded), shifts it to the observed frame using the provided redshift,
    and interpolates it onto the wavelength grid of the observed spectrum.
    It then identifies wavelength regions within the Lyman-Werner band
    where absorption features occur.

    Two masks are produced:
    - `mask_data`: pixels corresponding to any absorption feature.
    - `mask_core`: pixels corresponding to the core of strong absorption
      features.

    :param wavelength: The wavelength grid of the observed spectrum.
    :type wavelength: `np.ndarray`
    :param redshift: The redshift of the observed spectrum.
    :type redshift: `float`
    :param profile: A list containing the wavelength and flux arrays of the synthetic H₂ profile.
    :type profile: `list[np.ndarray, np.ndarray]`
    :param mask: The mask array for the observed spectrum, provided by the SPARCL database.
    :type mask: `np.ndarray`

    :return: A tuple containing the two masks (`mask_data` and `mask_core`) and the rebinned 
    synthetic H₂ profile.
    :rtype: `tuple[np.ndarray, np.ndarray, np.ndarray]`
    """

    # ==============
    # Initialization
    # ==============

    # Retrieve the wavelength and flux arrays for the current synthetic H₂ profile from the class
    # attribute `_h2_profiles` that was loaded into memory
    h2_synthetic_wavelength, h2_synthetic_flux = profile.wavelength, profile.flux

    # Shift the synthetic H₂ profile to the current redshift value to account for the 
    # cosmological redshift of the quasar and its potential associated absorption features
    h2_synthetic_wavelength_obs = h2_synthetic_wavelength * (1. + redshift)

    # Rebining the shifted synthetic H₂ profile to the observed wavelength grid of the spectrum
    interpolation_function = interp1d(h2_synthetic_wavelength_obs, h2_synthetic_flux, bounds_error=False, fill_value=0.)
    h2_synthetic_flux_rebinned = interpolation_function(wavelength)

    # Determine the wavelength range for the current redshift value
    min_wavelength = H2_LYMAN_WERNER_BANDS[0]*(1. + redshift)
    max_wavelength = H2_LYMAN_WERNER_BANDS[1]*(1. + redshift)

    # Selcting pixels in the Lyman-Werner region where there is an absorption feature in the 
    # synthetic H₂ profile (i.e., where the flux is below `ABSORPTION_FEATURE_THRESHOLD`)
    mask_data = (wavelength >= min_wavelength) & (wavelength <= max_wavelength) & (h2_synthetic_flux_rebinned < ABSORPTION_FEATURE_THRESHOLD)

    # Same selction but for the core absorption features in the synthetic H₂ profile (i.e.,
    # where the flux is below `CORE_ABSORPTION_FEATURE_THRESHOLD`)
#    min_flux = np.min(h2_synthetic_flux)
#    core_threshold = min_flux + CORE_ABSORPTION_FEATURE_THRESHOLD * (1 - min_flux)
#    mask_core = (wavelength >= min_wavelength) & (wavelength <= max_wavelength) & (h2_synthetic_flux_rebinned < core_threshold)

    # ==============
    # Peak detection for core transmission mask
    # ==============

    # Find the peaks in the synthetic H₂ profile using the scipy module
    peaks, _ = find_peaks(1-h2_synthetic_flux_rebinned[mask_data])
    # Convert the indices to the observed wavelength grid
    peaks = np.where(mask_data)[0][peaks]
    # Initialize a list to store the selected peaks and the current group, to only keep the 
    # deepest absorption features of each groups
    selected_peaks = []
    current_group = [peaks[0]]
    # Convert the absorption flux into a signal
    signal = 1 - h2_synthetic_flux_rebinned
    # Loop on found peaks
    for p in peaks[1:]:
        # If the peak is within 20 pixels of the previous peak, add it to the group
        if abs(p - current_group[-1]) < 20:
            current_group.append(p)
        # Else, select the peak with the highest signal
        else:
            best = max(current_group, key=lambda x: signal[x])
            selected_peaks.append(best)
            # Reset the current groupe
            current_group = [p]
    # Compute the peak with the highest signal of the last group
    best = max(current_group, key=lambda x: signal[x])
    selected_peaks.append(best)
    # Convert the list to an array and only keep the 6 deepest absorption features
    selected_peaks = np.array(selected_peaks)[:6]
    # If less than 6 peaks are found, print a warning message for degu purposes
    if len(selected_peaks) != 6:
        print(f"[DEBUG] Only {len(selected_peaks)} peaks found for redshift {redshift}")
        
    # Setting the width of the absorption feature to 10 pixels
    line_width = 20
    # Initiaizing the the mask for the core absorption features
    mask_core = np.zeros_like(wavelength, dtype=bool)
    # Computing core mask for each line
    for peak_idx in selected_peaks:
        # Computing the local mask
        local_mask = (wavelength >= wavelength[max(0, peak_idx - line_width)]) & (wavelength <= wavelength[peak_idx + line_width])
        # Retrieving the corresponding flux array and its minimum
        local_flux = h2_synthetic_flux_rebinned[local_mask]
        local_min = np.min(local_flux)
        # Defining the local threshold for the core absorption feature
        local_threshold = local_min + CORE_ABSORPTION_FEATURE_THRESHOLD * (1 - local_min)
        # Updating the core mask with the local mask where the flux is below the local threshold
        mask_core |= local_mask & (h2_synthetic_flux_rebinned < local_threshold)
    
    # ==============
    # Final masking
    # ==============

    # Retrieve the mask retrieved from the SPARCL database to the synthetic H₂ profile and the spectrum itself
    # and apply it to the other mask
    mask_pix_ok = mask == 0
    mask_data = np.logical_and(mask_data, mask_pix_ok)
    mask_core = np.logical_and(mask_core, mask_pix_ok)

    # Return the masks and the rebinned synthetic H₂ profile
    return mask_data, mask_core, h2_synthetic_flux_rebinned
