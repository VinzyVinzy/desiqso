"""
This module contains a class to store the results of the cross-correlation analysis.
It allows for easy loading and management of the results.
"""

# Importing necessary libraries
import numpy as np
import os
import pandas as pd

# Local imports
from src.desiqso.config import (PRELIMINARY_DATA_PATH, VISUAL_INSPECTION_PATH, MAGNITUDES_DATA_FOLDER, CROSS_CORRELATION_RESULTS_FOLDER, NEW_CANDIDATES_PATH, EXPECTED_CORE_TRANSMISSIONS_PATH, SNR_THRESHOLD,)
from src.desiqso.constants import (Categories, ColNames, Modes, VISUAL_LIST,)
from src.desiqso.utils.helpers import (parse_cell, compute_grade, compute_relative_speed, _is_valid,)

# Class to store the results of the cross-correlation analysis
class AnalysisResults:
    """
    This class stores the results of the cross-correlation analysis, as well
    as the preliminary analysis results, the expected core transmissions values
    for each synthetic profile, and the new candidates list.

    It currently contains the following class attributes:
    - _results: dict[str, pd.DataFrame]
    - _visual_inspection: pd.DataFrame
    - _preliminary_results: dict[str, pd.DataFrame]
    - _expected_cor_trans: dict[str, float]
    - _candidates: list[np.ndarray, np.ndarray]
    - _magnitudes: pd.DataFrame

    The available class methods are:
    - load_results(folder : str = CROSS_CORRELATION_RESULTS_FOLDER, verbose : bool = True)
    - reload(folder : str = CROSS_CORRELATION_RESULTS_FOLDER, verbose : bool = True)
    - load_preliminary_results(verbose : bool = True)
    - load_visual_inspection_results(verbose : bool = True)
    - load_expected_core_transmissions()
    - load_magnitudes(verbose : bool = True)
    - results_survey(mode:str = Modes.ALL, profile_name : str = "all", thresholds_dict:dict = {}) -> pd.DataFrame
    - export_results_list(path : str, mode : str, thresholds : dict[str, tuple[float]]))
    """

    # Class attribute to store the analysis results
    _results : pd.DataFrame = None
    # Class attribute to store the visual inspection results
    _visual_inspection : pd.DataFrame = None
    # Class attribute to store another analysis results
    _preliminary_results : dict[str, pd.DataFrame] = None
    # Class attribute to store the expected core transmissions
    _expected_cor_trans : dict[str, float] = None
    # Class attribute to store the candidates
    _candidates : list[np.ndarray, np.ndarray] = None
    # Class attribute to store the magnitudes data
    _magnitudes : pd.DataFrame = None

    # Class method to load the analysis results from local files
    @classmethod
    def load_results(cls, folder : str = CROSS_CORRELATION_RESULTS_FOLDER, verbose : bool = True) -> None:
        """
        This class method loads the cross-correlation analysis results from local files only once to save extra 
        time and guarantee easier access. Also loads the results table associated with the preliminary analysis and
        the low-SNR table. If available, it also loads the new candidates list and the expected core transmissions
        for the synthetic profiles. Returns to the main function if there are no results to load.

        :param folder: The folder containing the results of the cross-correlation analysis.
        :type folder: str
        :param verbose: Whether to print information about the loading process.
        :type verbose: bool
        """

        # If the results of the cross-correlation analysis are already loaded
        if cls._results is not None:
            # Return to the main program
            return

        
        # Initialize class attributes
        cls._results             : pd.DataFrame     = None
        cls._candidates          : dict[str, str]   = {}
        cls._low_snr             : pd.DataFrame     = None
        cls._failed              : pd.DataFrame     = None
        cls._magnitudes          : pd.DataFrame     = None

        # Inform user
        if verbose:
            print("\n[INFO] Loading preliminary analysis results...")
        # Loading preliminary analysis results from local files
        cls.load_preliminary_results(verbose=verbose)
        # Loading visual inspection results from local files
        cls.load_visual_inspection_results(verbose=verbose)
        # Loading the magnitudes data from a local file
        cls.load_magnitudes(verbose=verbose)

        # Inform user
        if verbose:
            print("[INFO] Loading cross-correlation analysis results...")

        # If the results of the cross-correlation analysis is available
        if os.path.exists(folder) :

            # Loop on the results tables of the cross-correlation analysis
            for result_table in [file for file in os.listdir(folder) if file.endswith(".npy") or file.endswith(".txt")]:
                # Loading analysis results as DataFrames for faster access
                results = pd.read_csv(folder+result_table, sep="\t")
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

    # Class method to forcefully reload the results of the cross-correlation analysis from another folder as the default one
    @classmethod
    def reload(cls, folder : str = CROSS_CORRELATION_RESULTS_FOLDER, verbose : bool = True) -> None:
        """
        This class method reloads the results of the cross-correlation analysis from another folder as the default one.

        :param folder: The folder containing the results of the cross-correlation analysis.
        :type folder: str
        :param verbose: If True, prints information about the cross-correlation analysis results.
        :type verbose: bool
        """
        # Reseting the class attribute
        cls._results = None
        # Loading the results of the cross-correlation analysis from the new folder
        cls.load_results(folder, verbose=verbose)

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
            cls._preliminary_results["unsure_candidates"] = pd.read_csv(PRELIMINARY_DATA_PATH+"unsure_candidates.txt", sep=r"\s+", header=None, names=col_names)
            cls._preliminary_results["confirmed_candidates"]  = pd.read_csv(PRELIMINARY_DATA_PATH+"confirmed_candidates.txt", sep=r"\s+", header=None, names=col_names+["#"])
            cls._preliminary_results["rejected_candidates"]   = pd.read_csv(PRELIMINARY_DATA_PATH+"rejected_candidates.txt", sep=r"\s+", header=None, names=col_names+["#"])
            
        # Inform user
        if verbose:
            print("[INFO] Preliminary analysis results loaded.\n")

        # Return to the main programm
        return

    # Class method to load the results of the visual inspection
    @classmethod
    def load_visual_inspection_results(cls, verbose : bool = True) -> None:
        """Class method to load the results of the visual inspection from a local file.
        
        The function first checks if the visual inspection results are already loaded to save time.
        :type verbose: bool
        """
        # If the results of the visual inspection are available
        if os.path.exists(VISUAL_INSPECTION_PATH):
            # Defining the columns names
            col_names = [ColNames.FILENAME, ColNames.CATEGORY]
            # Reading the file
            cls._visual_inspection = pd.read_csv(f"{VISUAL_INSPECTION_PATH}selection.dat", sep=r"\s+", header=None, names=col_names)

        # Inform user
        if verbose:
            print("[INFO] Visual inspection results loaded.\n")

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

    # Class method to load the magnitudes data from a local file and store it in a DataFrame for faster access
    @classmethod
    def load_magnitudes(cls, verbose : bool = True) -> None:
        """
        This class method loads the magnitudes data from a local file and stores it in a DataFrame for faster access.
        If the magnitudes data is not available, it returns to the main program with a warning message.

        :param verbose: If True, it prints a message to the user. Defaults to True.
        :type verbose: bool
        :returns None: This function does not return anything.
        """

        # Checking if the magnitudes folder is empty
        if len(os.listdir(MAGNITUDES_DATA_FOLDER)) == 0:
            # If it is empty, return to the main program with a warning message
            print("[WARNING] No magnitudes data available. Please run the `make download` command to retrieve the magnitudes data and save it in a local file.")
            return
        
        # If the magnitudes data is already loaded, return to the main program to save time
        if cls._magnitudes is not None:
            return
        
        # Load the magnitudes data from a local file and store it in a DataFrame for faster access
        cls._magnitudes = pd.read_csv(f"{MAGNITUDES_DATA_FOLDER}mag_table.csv") 
        # Inform user
        if verbose:
            print("[INFO] Magnitudes data loaded.\n")

        # Return to the main program
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
            ColNames.QSO_Z        :   thresholds_dict.get(ColNames.QSO_Z, (None, None)),
            ColNames.SNR          :   thresholds_dict.get(ColNames.SNR, (SNR_THRESHOLD, None)),
            ColNames.CNR          :   thresholds_dict.get(ColNames.CNR, (None, None)),
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
        if mode in VISUAL_LIST:
            table = table[table[ColNames.CATEGORY] == mode]
        elif mode == Modes.NEW:
            table = table[table[ColNames.FILENAME].isin(set(cls._candidates.keys()))]
        elif mode == Modes.VISUAL:
            table = table[table[ColNames.CATEGORY].isin(VISUAL_LIST)]
        elif mode == Modes.RANDOM:
            table = table.sample(n=100, axis=1)
        elif mode == Modes.OTHER:
            table = table[table[ColNames.CATEGORY] == Categories.OTHER]
        elif mode == Modes.VALID:
            table = table[table[ColNames.IS_VALID] == 1]
        elif mode == Modes.CANDIDATES:
            table = table[table[ColNames.CATEGORY].isin([Categories.CONFIRMED, Categories.UNSURE])]

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

    # Class method to export a subset of the results of the cross-correlation analysis
    @classmethod
    def export_results_list(cls, path : str, mode : str, thresholds : dict[str, tuple[float]]) -> None:
        """
        This class method allows the user to export a subset of the results of the cross-correlation analysis.

        :param path: The path to the output file.
        :type path: str
        :param mode: The mode of the survey. It can be any value from the `Modes` Enum.
        :type mode: str
        :param thresholds: A dictionary containing the thresholds for each parameter.
        :type thresholds: dict[str, tuple[float]]
        :returns None: This method does not return anything.
        """

        # Retrieving the results corresponding to the selected mode and thresholds
        results_table = cls.results_survey(mode=mode, thresholds_dict=thresholds)
        # Creating the directory if it does not exist
        savepath = os.path.dirname(path)
        os.makedirs(savepath, exist_ok=True)
        # Sort the results table by correlation parameter
        results_table = results_table.sort_values(by=ColNames.CORR_PARAM, ascending=False)
        # Selecting the columns to keep from the results table
        columns_to_keep = [ColNames.NAME, ColNames.QSO_Z, ColNames.Z, ColNames.CORR_COEFF, ColNames.CORE_TRANS, ColNames.CORR_PARAM, ColNames.SNR, ColNames.CNR, ColNames.GRADE, ColNames.REL_SPEED, ColNames.CATEGORY]
        # Exporting the results to a CSV file
        results_table[columns_to_keep].to_csv(path+".csv", index=False, float_format='%.5f')
        # Returning to the main function
        return

# Utility function to identify the category (data group) of a spectrum
def which_data_group(filename: str) -> str:
        """
        Identify the category (data group) of a spectrum using the visual inspection results.

        :param filename: The name of the spectrum file.
        :type filename: str

        :return: The category (data group) of the spectrum.
        :rtype: str
        """
        # Retrieve the row corresponding to the filename passed as argument
        row = AnalysisResults._visual_inspection[AnalysisResults._visual_inspection[ColNames.FILENAME] == filename]

        # If the row is empty, return "OTHER"
        if row.empty:
            return Categories.OTHER

        # Associating the category (data group) to the spectrum
        match row[ColNames.CATEGORY].iloc[0]:
            # If the spectrum is confirmed
            case "SELECT":
                return Categories.CONFIRMED
            # If the spectrum is unsure
            case "UNSURE":
                return Categories.UNSURE
            # If the spectrum is rejected
            case "REJECT":
                return Categories.REJECTED
            