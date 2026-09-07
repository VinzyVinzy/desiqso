"""
This module contains the entry point for the `physical-analysis` command, which is 
responsible for performing the physical analysis of the quasar absorption systems. 

The analysis includes plotting the distribution of categories of systems, as well as 
plotting various color graphs to investigate the physical properties of the systems as 
a function of different parameters.
"""

# Local imports
from src.desiqso.constants import ColNames
from src.desiqso.analysis.physical_analysis import (plot_categories_dist, plot_color_graphs, plot_morph_analysis,)

# Entry point for the `make sample-statistics` command
if __name__ == "__main__":

    # Calling the function to plot the statistics of the sample of spectra used for the analysis
    plot_categories_dist()
    
    # Calling the function to plot the graphics related to the color of the systems as a function of different parameters
    plot_color_graphs(x_cols = [ColNames.QSO_Z])

    # Calling the function to plot the morphology analysis of the systems
    plot_morph_analysis()
