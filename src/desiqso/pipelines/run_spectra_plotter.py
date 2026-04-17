"""
This module contains the entry point for the `make plot-spectra` command.
"""

# Local imports
from src.desiqso.constants import (ColNames, Modes,)
from src.desiqso.visualization.spectra import plot_spectra

# Entry point for the `make plot-spectra` command
if __name__ == "__main__":

    # Defining thresholds dictionary
    thresholds = {
        ColNames.CORR_PROB  :   (None, None),
        ColNames.CORE_TRANS :   (None, None),
        ColNames.CORR_COEFF :   (None, None),
        ColNames.CORR_PARAM :   (None, None),
        ColNames.Z          :   (None, None),
        ColNames.SNR        :   (None, None),
        ColNames.GRADE      :   (None, None),
        ColNames.REL_SPEED  :   (None, None),
    }

    # Calling function to plot all spectra
    plot_spectra(mode=Modes.ALL, thresholds_dict=thresholds)
