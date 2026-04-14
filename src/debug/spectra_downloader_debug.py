"""
This module contains functions that have debugging purposes. It currently contains a function to check that
all the quasar spectra are correctly downloaded, without any missing fields.
"""

# Package imports
from astropy.io import fits
import os
from tqdm import tqdm

# Local imports
from src.desiqso.config import SPECTRA_DATA_FOLDER

# Function to check that all the quasar spectra are correctly downloaded
def check_spectra_downloaded() -> None:
    """
    This function checks that all the quasar spectra are correctly downloaded, without any missing fields.
    It loops on all the `.fits` files in the spectra data folder, checks each file for missing fields and
    prints the results in a easy-to-read format.
    """

    # Inform user
    print("\n[INFO] Checking that all the quasar spectra are correctly downloaded...")

    # Initializing dictionnary containing the list of missing fields for each spectrum file
    bad_files = {}

    # Loop on the spectra files
    for filename in tqdm([file for file in os.listdir(SPECTRA_DATA_FOLDER) if file.endswith(".fits")], desc="Checking spectra", unit="spectra"):
        # Creating the file path to the `.fits` file
        file_path = os.path.join(SPECTRA_DATA_FOLDER, filename)
        # Checking if there are missing fields in the `.fits` file
        missing = check_fits_file(file_path)
        # If there are missing fields, add them to the dictionnary
        if missing:
            bad_files[filename] = missing
    
    # Inform user
    print("\n[INFO] All the spectra were analyzed! Printing the results...")
     
    # Loop on the issues found in the dictionnary
    for filename, issues in bad_files.items():
        # Print the filename and the list of issues
        print(f"\nFile {filename} has the following issues:")
        # Loop on the issues
        for issue in issues:
            # Print the issue
            print(f"\t- {issue}")

# Function to check that a single `.fits` file has all the required fields
def check_fits_file(file_path : str) -> list[str]:
    """
    This function checks that a single `.fits` file has all the required fields. It checks the data fields, 
    the header keys, and the extensions of the `.fits` file. If any of these are missing, it adds them to the 
    list of missing fields. If the file cannot be opened, it adds a missing field to the list. It returns the 
    list of missing fields.

    :param file_path: The path to the `.fits` file to check.
    :type file_path: str
    :return: The list of missing fields in the `.fits` file.
    :rtype: list[str]
    """

    # Initialize the list of missing fields
    missing = []

    # Try to open the file and check if all the required fields are present
    try:
        # Opening the file using `astropy.io.fits` module
        with fits.open(file_path) as hdul:
            # Retrieving the data and header of the `.fits` file
            data = hdul[1].data
            hdr = hdul[0].header

            # Checking if the data fields are available
            for i in [0,1,2]:
                # Trying to access the data field
                try:
                    _ = data.field(i)
                # If the field is not available, add it to the list of missing fields
                except Exception:
                    missing.append(f"data.field({i})")
            
            # Defining the list of header keys to retrieve
            header_keys = ["Z_EM", "Z_ERR", "Z_WARN", "NAME", "DATAREL", "SPECTYPE", "SPECID", "RA", "DEC", "CHI_2", "TSNR2"]

            # Loop on the header keys
            for key in header_keys:
                # If the key is not in the header, add it to the list of missing fields
                if key not in hdr:
                    missing.append(f"header:{key}")
            
            # Loop on the extensions
            for extension in ["MODEL", "MASK"]:
                # If the extension is not in the file, add it to the list of missing fields
                if extension not in hdul:
                    missing.append(f"extension:{extension}")
    
    # If the file cannot be opened, add a missing field
    except Exception:
        missing.append("corrupted_or_unreadable")
    
    # Return the list of missing fields
    return missing
