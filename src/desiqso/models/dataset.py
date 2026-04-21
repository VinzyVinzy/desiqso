"""
This module contains a class to store the results of the cross-correlation analysis.
It allows for easy loading and management of the results.
"""

# Importing necessary libraries
import numpy as np
import os
import pandas as pd

# Local imports
from src.desiqso.config import (PRELIMINARY_DATA_PATH, CROSS_CORRELATION_RESULTS_FOLDER, NEW_CANDIDATES_PATH, EXPECTED_CORE_TRANSMISSIONS_PATH, SNR_THRESHOLD,)
from src.desiqso.constants import (Categories, ColNames, Modes, PREL_LIST,)
from src.desiqso.utils.helpers import (parse_cell, compute_grade, compute_relative_speed, _is_valid,)

# Class to store the results of the cross-correlation analysis
class AnalysisResults:
    """
    This class stores the results of the cross-correlation analysis, as well
    as the preliminary analysis results, the expected core transmissions values
    for each synthetic profile, and the new candidates list.

    It currently contains the following class attributes:
    - _results: dict[str, pd.DataFrame]
    - _preliminary_results: dict[str, pd.DataFrame]
    - _expected_cor_trans: dict[str, float]
    - _candidates: list[np.ndarray, np.ndarray]

    The available class methods are:
    - load_results()
    - load_preliminary_results(verbose : bool = True)
    - load_expected_core_transmissions()
    """

    # Class attribute to store the analysis results
    _results : pd.DataFrame = None
    # Class attribute to store another analysis results
    _preliminary_results : dict[str, pd.DataFrame] = None
    # Class attribute to store the expected core transmissions
    _expected_cor_trans : dict[str, float] = None
    # Class attribute to store the candidates
    _candidates : list[np.ndarray, np.ndarray] = None

    # Class method to load the analysis results from local files
    @classmethod
    def load_results(cls, verbose : bool = True) -> None:
        """
        This class method loads the cross-correlation analysis results from local files only once to save extra 
        time and guarantee easier access. Also loads the results table associated with the preliminary analysis and
        the low-SNR table. If available, it also loads the new candidates list and the expected core transmissions
        for the synthetic profiles.
        
        Exits the program if there are no results to load.
        """

        # If the results of the cross-correlation analysis are already loaded
        if cls._results is not None:
            # Return to the main program
            return

        # Inform user
        if verbose:
            print("\n[INFO] Loading preliminary analysis results...")
        # Loading preliminary analysis results from local files
        cls.load_preliminary_results(verbose=verbose)

        # Initialize class attributes
        cls._results             : pd.DataFrame     = None
        cls._candidates          : dict[str, str]   = {}
        cls._low_snr             : pd.DataFrame     = None
        cls._failed              : pd.DataFrame     = None

        # Inform user
        if verbose:
            print("[INFO] Loading cross-correlation analysis results...")

        # If the results of the cross-correlation analysis is available
        if os.path.exists(CROSS_CORRELATION_RESULTS_FOLDER) :

            # Loop on the results tables of the cross-correlation analysis
            for result_table in [file for file in os.listdir(CROSS_CORRELATION_RESULTS_FOLDER) if file.endswith(".npy") or file.endswith(".txt")]:
                # Loading analysis results as DataFrames for faster access
                results = pd.read_csv(CROSS_CORRELATION_RESULTS_FOLDER+result_table, sep="\t")
                # Cleaning the column names
                results.columns = results.columns.str.replace("# ", "", regex=False)
                # Parsing the cells to obtain the right data type for the column "Best fit J core transmission"
                if ColNames.J_CORE_TRANS in results.columns:
                    # Applying the `parse_cell` function to convert the cell to a list
                    results[ColNames.J_CORE_TRANS] = results[ColNames.J_CORE_TRANS].apply(parse_cell)
                    # Applying the `compute_grade` function to obtain the grade directly in the DataFrame
                    results[ColNames.GRADE] = results[ColNames.J_CORE_TRANS].apply(compute_grade)
                    # Applying the `_is_valid` function to obtain the validity directly in the DataFrame
                    results[ColNames.IS_VALID] = results.apply(lambda row: _is_valid(row), axis=1)
                # Applying the `which_data_group` function to obtain the data group directly in the DataFrame
                results[ColNames.CATEGORY] = results[ColNames.FILENAME].apply(which_data_group)
                # Applying the `compute_relative_speed` function to obtain the relative speed directly in the DataFrame
                if ColNames.Z in results.columns:
                    # Applying the `compute_relative_speed` function to obtain the relative speed directly in the DataFrame
                    results[ColNames.REL_SPEED] = results.apply(lambda row: compute_relative_speed(row[ColNames.Z], row[ColNames.QSO_Z]), axis=1)
                # If the file is the low-SNR table
                if "low_snr" in result_table:
                    cls._low_snr = results
                # If the file is the failed table
                elif "failed" in result_table:
                    cls._failed = results
                # If it is an analysis results table
                else:
                    # Adding the name of the synthetic profile
                    results[ColNames.PROFILE] = result_table[:-4]
                    # Storing the results
                    cls._results = results

            # Inform user
            if verbose:
                print("[INFO] Cross-correlation analysis results loaded.\n")

            # Class method to load the expected core transmissions from local file
            #cls.load_expected_core_transmissions()
            
            # If the file containing the new candidates is available
            if os.path.exists(NEW_CANDIDATES_PATH):
                # Reading the expected core transmissions file
                with open(NEW_CANDIDATES_PATH, "r") as file:
                    # Loop on the lines
                    for line in file:
                        # Split the line to retrieve the name and the expected core transmission
                        filename, profile_name, _ = line.strip().split("\t")
                        # Storing the expected core transmission
                        cls._candidates[filename] = profile_name
                # Inform user
                if verbose:
                    print("[INFO] New candidates loaded.")

            # Return to the main programm
            return
        
        # If the results of the cross-correlation analysis is not available
        else :
            # Inform user
            print("[ERROR] No results of the cross-correlation analysis available. Please run the " \
            "`make run` command to execute cross-correlation analysis before plotting the spectra.")
            # Exit the programm
            os._exit(1)

    # Class method to load the results of the preliminary analysis
    @classmethod
    def load_preliminary_results(cls, verbose : bool = True) -> None:
        """
        Class method to load the results of the preliminary analysis results from local files, and stores them in a 
        dictionary for faster access.

        :param verbose: If True, prints information about the preliminary analysis results.
        :type verbose: bool
        """
        
        # Initialise class attribute
        cls._preliminary_results : dict[str, pd.DataFrame] = {}
        # If the results of the preliminary analysis are available
        if os.path.exists(PRELIMINARY_DATA_PATH):
            # Loading preliminary analysis results as arrays for faster access
            col_names = [ColNames.FILENAME, ColNames.NAME, ColNames.RA, ColNames.DEC, ColNames.QSO_Z, ColNames.Z, ColNames.CORR_COEFF, ColNames.CORR_PROB, ColNames.CORE_TRANS, ColNames.SNR]
            cls._preliminary_results["borderline_candidates"] = pd.read_csv(PRELIMINARY_DATA_PATH+"borderline_candidates.txt", sep=r"\s+", header=None, names=col_names)
            cls._preliminary_results["confirmed_candidates"]  = pd.read_csv(PRELIMINARY_DATA_PATH+"confirmed_candidates.txt", sep=r"\s+", header=None, names=col_names+["#"])
            cls._preliminary_results["rejected_candidates"]   = pd.read_csv(PRELIMINARY_DATA_PATH+"rejected_candidates.txt", sep=r"\s+", header=None, names=col_names+["#"])
            
        # Inform user
        if verbose:
            print("[INFO] Preliminary analysis results loaded.\n")

        # Return to the main programm
        return

    # Class method to load the expected core transmissions for each synthetic profile
    @classmethod
    def load_expected_core_transmissions(cls) -> None:
        """
        Class method to load the expected core transmissions for each synthetic profile from a local file.
        
        The function first checks if the expected core transmissions are already loaded to save time.
        Then, it checks if the file containing the expected core transmissions is available, and if it is, it reads the file 
        and stores the expected core transmissions in a dictionary.
        
        If the file is not available, the function prints an information message to the user and exits the program with an error.
        
        :return: This function does not return anything.
        :rtype: None
        """

        # If the expected core transmissions are already loaded, exit the function to save time
        if cls._expected_cor_trans is not None:
            return

        # Initialize class attribute        
        cls._expected_cor_trans  : dict[str, float] = {}

        # If the file containing the expected core transmissions is available
        if os.path.exists(EXPECTED_CORE_TRANSMISSIONS_PATH):
            # Reading the expected core transmissions file
            with open(EXPECTED_CORE_TRANSMISSIONS_PATH, "r") as file:
                # Loop on the lines
                for line in file:
                    # Split the line to retrieve the name and the expected core transmission
                    name, expected_core_transmission = line.strip().split("\t")
                    # Storing the expected core transmission
                    cls._expected_cor_trans[name] = float(expected_core_transmission)
            # Inform user
            print("[INFO] Expected core transmissions loaded.")

        # If the file containing the expected core transmissions is not available
        else:
            # Inform user
            print("[INFO] Expected core transmissions not loaded. Please run the `make dependencies-analysis` command to compute the expected core transmissions and save them.")
            # Exit the programm with an error
            os._exit(1)
        
        # Return to the main programm
        return

    # Class method to fast access to the pd.DataFrame containing the results of the cross-correlation analysis based on some selection parameters
    @classmethod
    def results_survey(cls, mode:str = Modes.ALL, profile_name : str = "all", thresholds_dict:dict = {}) -> pd.DataFrame:
        """
        This class method allows the user to fast access to the `pd.DataFrame` containing the results of the 
        cross-correlation analysis based on some selection parameters.

        :param mode: The mode of the survey. It can be any value from the `Modes` Enum. Defaults to `Modes.ALL`.
        :type mode: str
        :param profile_name: The name of the synthetic profile. Defaults to "best", which corresponds to the best synthetic profile for each spectrum.
        :type profile_name: str
        :param thresholds_dict: A dictionary containing the thresholds for each parameter. Defaults to  an empty dictionary.
        :type thresholds_dict: dict
        """

        # Creating a dictionary to store the thresholds and make sure no invalid values are passed
        thresholds = {
            ColNames.CORR_PROB    :   thresholds_dict.get(ColNames.CORR_PROB, (None, None)),
            ColNames.CORE_TRANS   :   thresholds_dict.get(ColNames.CORE_TRANS, (None, None)),
            ColNames.CORR_COEFF   :   thresholds_dict.get(ColNames.CORR_COEFF, (None, None)),
            ColNames.CORR_PARAM   :   thresholds_dict.get(ColNames.CORR_PARAM, (None, None)),
            ColNames.Z            :   thresholds_dict.get(ColNames.Z, (None, None)),
            ColNames.SNR          :   thresholds_dict.get(ColNames.SNR, (SNR_THRESHOLD, None)),
            ColNames.GRADE        :   thresholds_dict.get(ColNames.GRADE, (None, None)),
            ColNames.REL_SPEED    :   thresholds_dict.get(ColNames.REL_SPEED, (None, None)),
        }

        # Copying the results table to avoid modifying it
        table = cls._results.copy()
        
        # If the profile name is "best"
        if profile_name == "best":
            table = table[table[ColNames.BEST_FIT_FLAG] == 1]
        # If the profile name is not "best", retrieve the table corresponding to the selected profile
        elif profile_name != "all":
            table = table[table[ColNames.PROFILE] == profile_name]

        # Retrieving mask corresponding to the category selected
        if mode in PREL_LIST:
            table = table[table[ColNames.CATEGORY] == mode]
        elif mode == Modes.NEW:
            table = table[table[ColNames.FILENAME].isin(set(cls._candidates.keys()))]
        elif mode == Modes.PRELIMINARY:
            table = table[table[ColNames.CATEGORY].isin(PREL_LIST)]
        elif mode == Modes.RANDOM:
            table = table.sample(n=100, axis=1)
        elif mode == Modes.VALID:
            table = table[table[ColNames.IS_VALID] == 1]

        # Initialize mask
        mask = pd.Series(True, index=table.index)
        # Loop on the thresholds dictionary items
        for key, (min_val, max_val) in thresholds.items():
            # Apply mask if a minimum value is given
            if min_val is not None:
                mask &= table[key] >= min_val
            # Apply mask if a maximum value is given
            if max_val is not None:
                mask &= table[key] <= max_val
        # Apply mask to the data and store the filtered data in a new DataFrame
        filtered_table = table[mask]

        # Returning the filtered results
        return filtered_table

# Utility function to identify the category (data group) of a spectrum
def which_data_group(filename: str) -> str:
        """
        Identify the category (data group) of a spectrum.
        """
        # Retrurn the category
        return Categories.CONFIRMED if filename in set(AnalysisResults._preliminary_results["confirmed_candidates"][ColNames.FILENAME]) else (Categories.BORDERLINE if filename in set(AnalysisResults._preliminary_results["borderline_candidates"][ColNames.FILENAME]) else (Categories.REJECTED if filename in set(AnalysisResults._preliminary_results["rejected_candidates"][ColNames.FILENAME]) else Categories.OTHER))
