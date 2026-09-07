"""
This module contains the entry point for the `make spectra-stacker` command.
"""

# Local imports
from src.desiqso.analysis.spectra_stacker import spectra_stacker
from src.desiqso.constants import (ColNames, Modes,)

# Entry point for the `make spectra-stacker` command
if __name__ == "__main__":

    # Defining thresholds dictionary
    thresholds = {
        ColNames.CORR_PROB  :   (None, None),
        ColNames.CORE_TRANS :   (None, None),
        ColNames.CORR_COEFF :   (None, None),
        ColNames.CORR_PARAM :   (None, None),
        ColNames.Z          :   (None, None),
        ColNames.QSO_Z      :   (None, None),
        ColNames.SNR        :   (None, None),
        ColNames.CNR        :   (None, None),
        ColNames.GRADE      :   (None, None),
        ColNames.REL_SPEED  :   (None, None),
    }

    # Calling function to plot all spectra
    spectra_stacker(mode=Modes.CONFIRMED, thresholds_dict=thresholds)
