"""
This module contains the entry point for the `run-mock-analysis` command.
"""

# Local imports
from src.desiqso.analysis.mocks_spectra import mock_analysis
from src.desiqso.models.profile import ProfileManager

# Entry point for the `run-mock-analysis` command
if __name__ == "__main__":

    # Load H2 profiles, if not already done
    ProfileManager.load_all()

    # Defining the profiles to use for the mock spectra analysis
    profile_to_add = ProfileManager.get(name="h2_profile_res-2650.0_ntot-19.5_J-0-1-2-3-4-5_Texc-75.0_b-5.0_pix-5.0")
    profile_to_fit = ProfileManager.get(name="h2_profile_res-2650.0_ntot-20.0_J-0-1_Texc-75.0_b-3.0_pix-5.0")

    # Calling the function to perform the complete mock analysis
    mock_analysis(profile_to_add, profile_to_fit)