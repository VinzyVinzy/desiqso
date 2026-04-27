"""
This module contains a class to store the results of the cross-correlation analysis for a spectrum record.
This class contains the methods to save the results of the cross-correlation analysis to a file.
"""

# Importing necessary libraries
from dataclasses import dataclass
import numpy as np
import os
from typing import Optional

# Local imports
from src.desiqso.config import (CROSS_CORRELATION_RESULTS_FOLDER, SYNTHETIC_PROFILES_FOLDER, USE_BASIC_SYNTHETIC_PROFILES,)
from src.desiqso.constants import ColNames
from src.desiqso.models.profile import Profile

# Class containing the results of the cross-correlation analysis for a spectrum record
@dataclass
class CrossCorrelationResult:
    """
    Class representing the results of the cross-correlation analysis for a spectrum record.
    
    It contains the following methods:
    - `initialize_results()`: Class method to initialize the output files for the spectra analysis results.
    - `save_to_file()`: Method to save the results of the cross-correlation analysis to a file.
    """

    # Class attributes representing the results of the cross-correlation analysis
    status:                     str
    file_name:                  Optional[str]               = None
    name:                       Optional[str]               = None
    ra:                         Optional[float]             = None
    dec:                        Optional[float]             = None
    redshift:                   Optional[float]             = None
    best_redshifts:             Optional[list[float]]       = None
    best_correlation_values:    Optional[list[float]]       = None
    best_correlation_probs:     Optional[list[float]]       = None
    best_core_transmissions:    Optional[list[float]]       = None
    best_J_core_transmissions:  Optional[list[list[float]]] = None
    best_correlation_parameters:Optional[list[float]]       = None
    snr:                        Optional[float]             = None
    continuum:                  Optional[float]             = None
    details:                    Optional[str]               = None
    best_fit_flags:             Optional[list[int]]         = None

    # Class attribute to initialize output files
    _results: Optional[dict[str, object]] = None

    # Class method to initialize output files for the spectra analysis results
    @classmethod
    def initialize_results(cls, output_folder : str = CROSS_CORRELATION_RESULTS_FOLDER, profiles_list = list[Profile]):
        """
        Class method to initialize the output files for the spectra analysis results.

        This method checks if the output files are not initialized yet. If not, it creates the results 
        folder if it does not exist, initializes the `_results` attribute and sets up the output files 
        for the cross-correlation analysis results of the spectra, including the output files for the 
        synthetic profiles and the low SNR spectra. It also defines the headers for the output files 
        and writes them to the files.

        :param output_folder: Folder to save the results.
        :type output_folder: str
        :param profiles_list: List of synthetic profiles used to perform the cross-correlation analysis.
        :type profiles_list: list[Profile]
        """
        
        # If output files are not initialized yet
        if cls._results is None:
            # If the results folder does not exist, create it
            os.makedirs(output_folder, exist_ok=True)

            # Initialize the `_results` attribute
            cls._results = {}

            # If the USE_BASIC_SYNTHETIC_PROFILES flag is True, setting up the output files
            if USE_BASIC_SYNTHETIC_PROFILES:
                cls._results["synth_a"] = os.path.join(output_folder, "synth_a.txt")
                cls._results["synth_b"] = os.path.join(output_folder, "synth_b.txt")

            # Setting up the output file for all the other synthetic profiles
            for profile in profiles_list:
                cls._results[profile.name] = os.path.join(output_folder, f"{profile.name}.npy")
            
            # Setting up the output file for low SNR spectra
            cls._results["low_snr"] = os.path.join(output_folder,"low_snr.txt")
            # Setting up the output file for the failed analyzed spectra
            cls._results["failed"] = os.path.join(output_folder,"failed.txt")

            # Initialize the `headers` dictionary
            headers = {}

            # Define headers for the output files
            for key in list(cls._results.keys())[:-1]:
                headers[key] = [
                        ColNames.FILENAME,  ColNames.NAME,          ColNames.RA,        ColNames.DEC,
                        ColNames.QSO_Z,     ColNames.Z,             ColNames.CORR_COEFF,ColNames.CORR_PROB,
                        ColNames.CORE_TRANS,ColNames.J_CORE_TRANS,  ColNames.CORR_PARAM,ColNames.SNR, 
                        ColNames.CONTINUUM, ColNames.BEST_FIT_FLAG,
                    ]
            # Define headers for the output file for low SNR spectra
            headers["low_snr"] = [
                    ColNames.FILENAME,  ColNames.NAME, ColNames.RA,         ColNames.DEC,
                    ColNames.QSO_Z,     ColNames.SNR,   ColNames.CONTINUUM,
                ]
            # Define headers for the output file for the failed analyzed spectra
            headers["failed"] = [
                    ColNames.FILENAME,  ColNames.NAME, ColNames.RA,         ColNames.DEC,
                    ColNames.QSO_Z,     ColNames.SNR,   ColNames.CONTINUUM, ColNames.DETAILS,
                ]

            # Write headers to the output files
            for key, path in cls._results.items():
                with open(path, "w") as file:
                    line = "# " + "\t".join(headers[key]) + "\n"
                    file.write(line)

    # Method to save the results in the corresponding output file based on the analysis status
    def save_results(self):
        """
        Function to save the results of the cross-correlation analysis in the corresponding output file 
        based on the analysis status (i.e., if the spectrum has low SNR or not). 
        
        The function writes the relevant information for each spectrum in a .txt format, including the 
        file name, name, coordinates, redshift, best fit redshift, best fit correlation value, 
        best fit correlation probability, best fit core transmission and SNR for spectra with SNR 
        above the defined threshold, and only the file name, name, coordinates, redshift and SNR for 
        spectra with low SNR.
        """
        # If the spectrum has low SNR, save the relevant information in the corresponding output file
        if self.status == "Low SNR":
            # Defining values to save
            values = [
                self.file_name,
                self.name,
                f"{self.ra:.6f}",
                f"{self.dec:.6f}",
                f"{self.redshift:.5f}",
                f"{self.snr:.2f}",
                f"{self.continuum:.2f}",
            ]
            # Saving results
            path = self.__class__._results["low_snr"]
            with open(path, "a") as file:
                file.write("\t".join(values) + "\n")
            # Returning
            return
        
        # Else, if the cross-correlation analysis was successful, save the relevant information in the corresponding output file
        elif self.status == "Success":
            # List of keys to the results files
            profile_keys = [k for k in self.__class__._results.keys() if k not in ["low_snr", "failed"]]
            # Loop on the synthetic profiles
            for profile_idx, key in enumerate(profile_keys):
                # Defining values to save
                values = [
                    self.file_name,
                    self.name,
                    f"{self.ra:.6f}",
                    f"{self.dec:.6f}",
                    f"{self.redshift:.5f}",
                    f"{self.best_redshifts[profile_idx]:.5f}",
                    f"{self.best_correlation_values[profile_idx]:.3f}",
                    f"{np.log10(self.best_correlation_probs[profile_idx]):.3f}",
                    f"{self.best_core_transmissions[profile_idx]:.3f}",
                    f"{self.best_J_core_transmissions[profile_idx]}",
                    f"{self.best_correlation_parameters[profile_idx]:.3f}",
                    f"{self.snr:.2f}",
                    f"{self.continuum:.2f}",
                    f"{self.best_fit_flags[profile_idx]}",
                ]
                # Saving results
                path = self.__class__._results[key]
                with open(path, "a") as file:
                    file.write("\t".join(values) + "\n")
        
        # Else, if the cross-correlation analysis failed, save the revelant information in the corresponding output file
        elif self.status == "Failed":
            # Defining values to save
            values = [
                self.file_name,
                self.name,
                f"{self.ra:.6f}",
                f"{self.dec:.6f}",
                f"{self.redshift:.5f}",
                f"{self.snr:.2f}",
                f"{self.continuum:.2f}",
                f"{self.details}",
            ]
            # Saving results
            path = self.__class__._results["failed"]
            with open(path, "a") as file:
                file.write("\t".join(values) + "\n")
            # Returning
            return
