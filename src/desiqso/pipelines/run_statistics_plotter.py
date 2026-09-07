"""
This module contains the entry point for the `plot-statistics` command.
It also defines the pairs of statistics to plot on the flight.
"""

# Local imports
from src.desiqso.constants import (ColNames, Modes,)
from src.desiqso.visualization.statistics import plot_statistics

# Entry point for `plot-statistics` command
if __name__ == "__main__":

    # Defining the pairs of statistics to plot
    plot_pairs = [
        (ColNames.SNR, ColNames.CORR_PARAM),
        (ColNames.QSO_Z, ColNames.CORR_PARAM),
        (ColNames.GRADE, ColNames.CORR_PARAM),
        (ColNames.CORR_COEFF, ColNames.CORE_TRANS),
        (ColNames.QSO_Z, ColNames.SNR),
        (ColNames.QSO_Z, ColNames.CNR),
        (ColNames.SNR, ColNames.CORE_TRANS),
        (ColNames.CNR, ColNames.CORE_TRANS),
        (ColNames.CNR, ColNames.SNR),
    ]

    # Defining thresholds dictionary
    thresholds = {
        ColNames.CORR_PROB  :   (None, None),
        ColNames.CORE_TRANS :   (None, None),
        ColNames.CORR_COEFF :   (None, None),
        ColNames.CORR_PARAM :   (None, None),
        ColNames.QSO_Z      :   (2.6, None),
        ColNames.Z          :   (None, None),
        ColNames.SNR        :   (3.0, None),
        ColNames.CNR        :   (None, None),
        ColNames.GRADE      :   (None, None),
        ColNames.REL_SPEED  :   (None, None),
    }

    # Calling function to plot the statistics of the analysis results
    plot_statistics(plot_pairs, thresholds, color_col=ColNames.QSO_Z, mode=Modes.ALL)
