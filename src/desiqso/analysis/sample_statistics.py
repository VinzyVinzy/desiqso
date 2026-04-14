""""""

# Packages import

# Local imports
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.models.profile import ProfileManager

# Function to plot the initial sample statistics
def plot_sample_statistics() -> None:
    """"""

    # Loading cross-correlation analysis results
    AnalysisResults.load_results()
    # Loading all the synthetic profiles
    ProfileManager.load_all()

    # 
    low_snr_table = AnalysisResults._low_snr.copy()
    failed_table = AnalysisResults._failed.copy()

    #
    