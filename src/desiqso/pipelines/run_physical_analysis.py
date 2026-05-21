""""""

# Local imports
from desiqso.constants import ColNames
from src.desiqso.analysis.physical_analysis import (plot_categories_dist, plot_color_graphs,)

# Entry point for the `make sample-statistics` command
if __name__ == "__main__":

    # Calling the function to plot the statistics of the sample of spectra used for the analysis
    plot_categories_dist()
    
    # Calling the function to plot the graphics related to the color of the systems as a function of different parameters
    plot_color_graphs(x_cols = [ColNames.QSO_Z])
