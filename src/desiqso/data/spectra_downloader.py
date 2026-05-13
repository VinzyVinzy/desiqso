"""
This module contains functions to download the preliminary analysis results spectra and all the available
spectra of QSOs with a redshift above 2.5 from the DESI-DR1 database.
"""

# Packages import
import math
import os
import pandas as pd
from sparcl.client import SparclClient
from tqdm import tqdm

# Local imports
from src.desiqso.config import NUMBER_OF_SPECTRA, REDSHIFT_RANGE, PRELIMINARY_DATA_PATH, SPECTRA_DATA_FOLDER
from src.desiqso.data.spectra_utils import process_spectrum_record

# Function retrieving all QSOs spectra from DESI-DR1 database, above a certain redshift
def retrieve_spectra_from_database(ra_list : list[int]) -> None:
    """
    This function retrieves all QSOs spectra from DESI-DR1 database, above a redshift of 2.5. It uses the
    `SparclClient` class to interact with the DESI-DR1 database and submit a query to retrieve the spectra
    corresponding to the selected criteria.

    :param ra_list: A list of Right Ascension (RA) ranges to retrieve the spectra from the DESI-DR1 database.
    :type ra_list: `list[int]`
    """
    
    # ===============
    # Query configuration
    # ================

    # Number of spectra to retrieve
    number_of_spectra = NUMBER_OF_SPECTRA

    # Constraints on spectra to retrieve
    constraints = {
        "spectype"      :   ["QSO"],
        "redshift"      :   REDSHIFT_RANGE,
        "data_release"  :   ["DESI-DR1"],
    }

    # Creation of Sparcl Client Instance
    client = SparclClient(read_timeout=60, announcement=False)
    # List of desired outfiels
    outfields = ["sparcl_id", "ra", "dec", "redshift", "spectype", "data_release"]

    # ================
    # Query execution
    # ================

    # Initialisation of retrieved record count
    count = 0
    # Loop on Right Ascension (RA)
    ra_step = 1    # RA step (in degrees)

    # Loop on RA slices
    for ra_start in tqdm(ra_list, desc="Scanning RA", unit="slice"):
        
        # Compute current RA range
        ra_end = ra_start + ra_step
        # Update constraints to fit the current RA
        constraints["ra"] = [ra_start, ra_end]
        # Inform user
        tqdm.write(f"\n[INFO] Retrieving spectra for RA range [{ra_start}, {ra_end}]")

        # Query execution
        found = client.find(constraints=constraints, outfields=outfields, limit=number_of_spectra)
        # Querry results user information
        tqdm.write(f"[INFO] Found {len(found.ids)} spectra for RA range [{ra_start}, {ra_end}]")

        # Fields to retrieve using the IDs found
        fields = ["desiname",   "ra",             "dec",              "sparcl_id",    "specid",   "flux",
                  "wavelength", "ivar",           "data_release",     "spectype",     "redshift", "model",
                  "chi2",       "redshift_err",   "redshift_warning", "mask",         "tsnr2_qso",]
        
        # Inform user
        tqdm.write(f"[INFO] Downloading {len(found.ids)} spectra for RA range [{ra_start}, {ra_end}]")
        # Retrieve spectra data using the IDs found
        retrieved = client.retrieve(uuid_list=found.ids, include=fields, dataset_list=["DESI-DR1"], limit=number_of_spectra)
        # Inform user of querry results
        tqdm.write(f"[INFO] Successfully retrieved {len(retrieved.records)} spectra for RA range [{ra_start}, {ra_end}]")

        # Proccess retrieved spectra data
        for record in tqdm(retrieved.records, desc="Processing records", unit="record"):
            # Update retrieved record count
            count += 1
            # Process the current record using the dedicated function `process_spectrum_record`
            process_spectrum_record(record, count)

    # Inform user of the end of the spectra retrieval and processing
    print(f"\n[INFO] Successfully retrieved and processed {count} spectra from the DESI-DR1 database.")

# Function to check if all the quasar spectra from the preliminary analysis are available and, if not, dwonload them
def download_preliminary_spectra() -> None:
    """
    Function downloading the preliminary analysis results spectra from the DESI-DR1 database, using a 
    `SparclClient` instance.

    It first loads the preliminary analysis results from local files, then loops over the tables of the 
    preliminary analysis and downloads the spectra of the corresponding QSOs in each table.

    This function is intended to be used before downloading the other spectra from the DESI-DR1 database.
    """
    
    # ===============
    # Configuration
    # ================
    
    # List of desired outfiels    
    outfields = ["sparcl_id", "ra", "dec", "redshift", "spectype", "data_release"]
    # Fields to retrieve using the IDs found
    fields = ["desiname",   "ra",             "dec",              "sparcl_id",    "specid",   "flux",
                "wavelength", "ivar",           "data_release",     "spectype",     "redshift", "model",
                "chi2",       "redshift_err",   "redshift_warning", "mask",         "tsnr2_qso",]
                
    # Creation of Sparcl Client Instance
    client = SparclClient(connect_timeout=60.,read_timeout=10000, announcement=False)
    
    # ===============
    # Preliminary analysis results loading
    # ===============

    # Initializing the dictionary of tables of the preliminary analysis
    tables_dict = {}
    # If the results of the preliminary analysis are available
    if os.path.exists(PRELIMINARY_DATA_PATH):
        # Loading preliminary analysis results as arrays for faster access
        tables_dict["unsure_candidates"] = pd.read_csv(PRELIMINARY_DATA_PATH+"unsure_candidates.txt", sep=r"\s+", header=None, names=["File Name", "Name", "RA (deg)", "DEC (deg)", "Redshift", "Best fit redshift", "Best fit correlation value", "Best fit correlation probability (log10)", "Best fit core transmission", "SNR"])
        tables_dict["confirmed_candidates"]  = pd.read_csv(PRELIMINARY_DATA_PATH+"confirmed_candidates.txt", sep=r"\s+", header=None, names=["File Name", "Name", "RA (deg)", "DEC (deg)", "Redshift", "Best fit redshift", "Best fit correlation value", "Best fit correlation probability (log10)", "Best fit core transmission", "SNR", "#"])
        tables_dict["rejected_candidates"]   = pd.read_csv(PRELIMINARY_DATA_PATH+"rejected_candidates.txt", sep=r"\s+", header=None, names=["File Name", "Name", "RA (deg)", "DEC (deg)", "Redshift", "Best fit redshift",	"Best fit correlation value", "Best fit correlation probability (log10)", "Best fit core transmission", "SNR", "#"])
    
    # ===============
    # Spectra downloading
    # ===============

    # Loop over the tables of the preliminary analysis
    for name, table in tables_dict.items():
        # Retrieving the list of files to download
        files_to_load=table["File Name"].tolist()
        
        # Loop over the files for each preliminary analysis result table
        for file in tqdm(files_to_load, desc=f"Downloading the {" ".join(name.split("_"))} spectra", unit="spectra"):
            
            # Check if the file is already downloaded
            if not os.path.exists(SPECTRA_DATA_FOLDER + f"{file}"):

                # Get the redshift and coordinates of the QSO using its filename in the preliminary analysis table
                row = table[table["File Name"]==file]
                z=row["QSO Redshift"].iloc[0]
                ra=row["RA (deg)"].iloc[0]
                dec=row["DEC (deg)"].iloc[0]
                qso_name=row["Name"].iloc[0]

                # Constraints on spectra to retrieve for the query
                constraints = {"spectype":["QSO"],
                            "redshift":[z-0.0005, z+0.0005],
                            "data_release":["DESI-DR1"],
                            "ra":[ra-0.0001, ra+0.0001],
                            "dec":[dec-0.0001, dec+0.0001],}
                
                # Query execution
                found = client.find(constraints=constraints, outfields=outfields, limit=1000)
                # Querry results user information
                tqdm.write(f"[INFO] Found {qso_name} spectrum. Downloading...")
                # Retrieve spectra data using the IDs found
                retrieved = client.retrieve(uuid_list=found.ids, include=fields, dataset_list=["DESI-DR1"], limit=1)
                # Inform user of querry results
                tqdm.write(f"[INFO] Successfully retrieved {len(retrieved.records)} spectra")
                # Process the current record using the dedicated function `process_spectrum_record`
                process_spectrum_record(retrieved.records[0], 0)
            
            # If the file is already downloaded
            else:
                # Inform user
                tqdm.write(f"[INFO] {file} already exists. Skipping...")

# Function to check that all the quasar spectra were correctly downloaded
def check_spectra_downloaded() -> list[int]:
    """
    This function checks that all the quasar spectra were correctly downloaded.
    It returns the list of RA slices that were not completely downloaded.

    :return list[int]: The list of RA slices that were not completely downloaded.
    """

    # Inform user
    print("\n[INFO] Checking that all the RA slices were correctly downloaded...")

    # Initializing the list of unique RA values
    all_ra = [0]

    # Initializing the counter of duplicate RA values
    ra_count = 0
    counters = []
    
    # Loop over the downloaded spectra files
    for spectrum_file in os.listdir(SPECTRA_DATA_FOLDER):

        # Retrieve RA coordinates from the file name
        ra = spectrum_file[6:15]
        # Convert RA coordinates into degrees
        hh = int(ra[0:2])
        mm = int(ra[2:4])
        ss = float(ra[4:9])
        ra_hours = hh + mm/60. + ss/3600.
        ra_deg = math.floor(ra_hours*15.)

        # If the RA value is not in the list of unique RA values
        if ra_deg not in all_ra:

            # If the previous RA slice was not completely downloaded
            if ra_count < 50:
                # Remove the previous RA slice from the list
                all_ra.pop()
                # Update the counter list
                counters.append(ra_count)

            # Add the current RA slice to the list
            all_ra.append(ra_deg)
            # Reset the counter
            ra_count = 0

        # If the RA value is in the list of unique RA values
        else:
            # Increment the counter
            ra_count += 1
    
    # Inform user
    print(f"Found {len(all_ra)} unique RA values.")

    # Check if all the RA values were found using a set
    all_ra_set = set(all_ra)
    expected = set(range(360))
    missing = sorted(expected - all_ra_set)

    # If all the RA values were found
    if len(missing) == 0:
        print("All RA values were found!\n")
        return []

    # Inform user
    print(f"Missing {len(missing)} RA values. List of missing (or uncorrectly downloaded) RA slices:\n{missing}\nAssociated counters:\n{counters}\n")
    # Return the list of missing RA slices
    return missing
