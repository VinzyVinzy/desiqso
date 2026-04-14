"""
This module contains the entry point for the `make dependencies-analysis` command.
"""

# Local imports
from src.desiqso.analysis.dependency_analysis import compute_profile_snr_dependency
from src.desiqso.models.profile import ProfileManager

# Entry point for `make dependencies-analysis` command
if __name__ == "__main__":

    # Load H2 profiles, if not already done
    ProfileManager.load_all()

    # Loop on all available synthetic profiles
    for profile in ProfileManager.all_profiles():

        # Calling the function to evaluate the dpendency of the correlation and core transmission with SNR
        compute_profile_snr_dependency(profile = profile, show=False, save=True)
    