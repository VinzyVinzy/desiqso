"""
This module is dedicated to perform tests on the code.
"""

# Packages import
import numpy as np
import os
import random

# Local imports
from src.desiqso.analysis.physical_analysis import residual_image_stacking_analysis
from src.desiqso.config import PAQS_SPECTRA_FOLDER
from src.desiqso.constants import (ColNames, Modes,)
from src.desiqso.models.spectrum import SpectrumRecord
from src.desiqso.models.dataset import AnalysisResults

# Function to perform tests
def test():

    AnalysisResults.load_results()
    print(np.median(AnalysisResults._results[ColNames.QSO_Z]))
    os._exit(1)

    # Defining thresholds dictionary
    thresholds = {
        ColNames.CORR_PROB  :   (None, None),
        ColNames.CORE_TRANS :   (None, None),
        ColNames.CORR_COEFF :   (None, None),
        ColNames.CORR_PARAM :   (None, None),
        ColNames.QSO_Z      :   (None, None),
        ColNames.Z          :   (None, None),
        ColNames.SNR        :   (None, None),
        ColNames.CNR        :   (None, None),
        ColNames.GRADE      :   (None, None),
        ColNames.REL_SPEED  :   (None, None),
    }

    residual_image_stacking_analysis(Modes.CANDIDATES, thresholds_dict=thresholds)

    return

# Entry point for the `make test` command
if __name__ == "__main__":

    # Calling the test function
    test()
