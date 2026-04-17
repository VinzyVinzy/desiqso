"""
This module contains the entry point for the `plot-statistics` command.
It also defines the pairs of statistics to plot.
"""

# Local imports
from src.desiqso.constants import (ColNames, Modes,)
from src.desiqso.visualization.statistics import plot_statistics

# Entry point for `plot-statistics` command
if __name__ == "__main__":

    # Defining the pairs of statistics to plot
    plot_pairs = [
#        (ColNames.CORR_PARAM, ColNames.CORE_TRANS),
#        (ColNames.SNR, ColNames.QSO_Z),
#        (ColNames.GRADE, ColNames.CORR_PARAM),
#        (ColNames.CORR_PARAM, ColNames.GRADE),
#        (ColNames.QSO_Z, ColNames.Z),
        (ColNames.CORR_COEFF, ColNames.CORE_TRANS),
    ]

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

    # Calling function to plot the statistics of the analysis results
    plot_statistics(plot_pairs, thresholds, mode=Modes.ALL)
