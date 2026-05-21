"""
This module contains the entry point for the `export-results` command, which is responsible for 
exporting the analysis results to a CSV file. The exported results are filtered based on specified 
thresholds for various parameters, such as correlation probability, core transition, correlation 
coefficient, correlation parameter, quasar redshift, absorber redshift, signal-to-noise ratio, 
continuum-to-noise ratio, grade, and relative speed. 

The exported file can be used for further analysis or sharing with collaborators.
"""

# Local imports
from src.desiqso.config import EXPORTED_LISTS_FOLDER
from src.desiqso.constants import (ColNames, Modes,)
from src.desiqso.models.dataset import AnalysisResults

# Entry point for `export-results` command
if __name__ == "__main__":
    
    # Loading the analysis results
    AnalysisResults.load_results()

    # Defining thresholds dictionary
    thresholds = {
        ColNames.CORR_PROB  :   (None, None),
        ColNames.CORE_TRANS :   (None, None),
        ColNames.CORR_COEFF :   (None, None),
        ColNames.CORR_PARAM :   (0.42, None),
        ColNames.QSO_Z      :   (2.6, None),
        ColNames.Z          :   (None, None),
        ColNames.SNR        :   (3.0, None),
        ColNames.CNR        :   (None, None),
        ColNames.GRADE      :   (None, None),
        ColNames.REL_SPEED  :   (None, None),
    }

    # Inform user
    print("[INFO] Exporting analysis results...")

    # Defining the path where the analysis results will be exported
    filepath = f"{EXPORTED_LISTS_FOLDER}new_candidates_42"
    # Calling function to export the analysis results
    AnalysisResults.export_results_list(path=filepath,mode=Modes.VALID, thresholds=thresholds)

    # Inform user
    print(f"[INFO] Analysis results exported successfully! Check the file: {filepath}.csv")
