"""
This module contains the entry point for the `run-mock-analysis` command.
It aims to run the completeness analysis on the flight using a synthetic H₂ profile
in order to mesure the performances of the algorithm.
"""

# Local imports
from src.desiqso.analysis.completeness import run_completeness_analysis
from src.desiqso.models.profile import Profile
from src.desiqso.constants import DESI_RESOLUTION_POWER

# Entry point for the `run-mock-analysis` command
if __name__ == "__main__":

    # Creating the synthetic profile to use for the completeness analysis on the flight
    profile_to_fit = Profile.from_synthetic(
        resolution_power= DESI_RESOLUTION_POWER,# Resolution power of the synthetic profile
        pixel_size      = 5.,                   # Pixel size in km/s for the synthetic profile
        T_exc0          = 75.,                  # Excitation temperature of the J=0 level in K
        Jmax            = 1,                    # Maximum rotational level to include in the synthetic profile (0 and 1 by default)
        Ntot            = 10**20.0,             # Total column density of H₂ in cm^-2
        b_param         = 3.,                   # Doppler parameter for the synthetic profile in km/s
        save            = False,                # Save the synthetic profile so the program can access it
        verbose         = False                 # Do not print information about the synthetic profile
    )

    # Saving the generated synthetic H₂ profile in the `synthetic_profiles` folder
    profile_to_fit.save()

    # Calling the function to run the completeness analysis
    run_completeness_analysis(profile_to_fit=profile_to_fit)
