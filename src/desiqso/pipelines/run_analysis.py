"""
This module contains the entry point for the `run-analysis` command. It also defines the mode of the analysis.
"""

# Local imports
from src.desiqso.analysis.cross_correlation import run_cross_correlation_analysis
from src.desiqso.analysis.new_candidates import find_new_candidates
from src.desiqso.constants import Modes

# Entry point for the `run-analysis` command
if __name__ == "__main__":

    # Calling the function to perform the cross-correlation analysis with the selected mode
    run_cross_correlation_analysis(mode=Modes.RANDOM)

    # Finding and saving the new candidates found
    find_new_candidates()
