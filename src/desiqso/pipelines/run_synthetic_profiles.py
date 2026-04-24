"""
This module contains the entry point for the `make generate-synthetic-profiles` command.
"""

# Local imports
from src.desiqso.constants import DESI_RESOLUTION_POWER
from src.desiqso.models.profile import Profile

# Entry point for `make generate-synthetic-profiles` command
if __name__ == "__main__":

    # Calling the function to generate the synthetic H₂ profile
    new_profile = Profile.from_synthetic(
        resolution_power= DESI_RESOLUTION_POWER,# Resolution power of the synthetic profile
        pixel_size      = 5.,                   # Pixel size in km/s for the synthetic profile
        T_exc0          = 75.,                  # Excitation temperature of the J=0 level in K
        Jmax            = 1,                    # Maximum rotational level to include in the synthetic profile (0 and 1 by default)
        Ntot            = 1e19,                 # Total column density of H₂ in cm^-2
        b_param         = 3.,                   # Doppler parameter for the synthetic profile in km/s
    )
    # Saving the generated synthetic H₂ profile in the `synthetic_profiles` folder
    new_profile.save()