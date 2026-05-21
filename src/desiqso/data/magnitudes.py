"""
This module contains the functions related to the download of the magnitudes data for the quasar spectra from the DESI-DR1 photometric database using TAP queries.
"""

# Package imports
from astroquery.utils.tap.core import TapPlus
import os
import pandas as pd
from sparcl.client import SparclClient
from tqdm import tqdm

# Local imports
from src.desiqso.config import (MAGNITUDES_DATA_FOLDER, REDSHIFT_RANGE, NUMBER_OF_SPECTRA,)
from src.desiqso.constants import MagColNames
from src.desiqso.data.spectra_utils import generate_spectrum_name

# Function to download the magnitudes data for the quasar spectra from the DESI-DR1 database using TAP queries and the `SparclClient()`
def download_magnitudes() -> None:
    """
    This function downloads the magnitudes data for the quasar spectra from the DESI-DR1 database using TAP queries and the `SparclClient()`.
    It retrieves the Target ID for each quasar and then performs a TAP query to retrieve the magnitudes data for each quasar using its Target ID.
    The retrieved magnitudes data is then saved in a local file in CSV format in the `MAGNITUDES_DATA_FOLDER` folder.
    """

    # Inform user of the magnitudes download process
    print("\n[INFO] Downloading magnitudes data from the DESI-DR1 database...")

    # Creation of a TapPlus instance to perform TAP queries on the DESI-DR1 database
    tap = TapPlus(url="https://datalab.noirlab.edu/tap")

    # Configuration of the base TAP query 
    query = """
    SELECT column_name, datatype, description
    FROM tap_schema.columns
    WHERE table_name = 'desi_dr1.photometry'
    """

    # List of columns to retrieve from the desi_dr1.photometry table
    col_to_retrieve = ["targetid", "dec",      "ra",      "flux_g",  "flux_r",             "flux_z",             "mw_transmission_g",  "mw_transmission_r", "mw_transmission_z",
                       "flux_w1",  "flux_w2",  "flux_w3", "flux_w4", "mw_transmission_w1", "mw_transmission_w2", "mw_transmission_w3", "mw_transmission_w4","gaia_phot_g_mean_mag", 
                       "gaia_phot_bp_mean_mag", "gaia_phot_rp_mean_mag"]
    # Converting the list of columns to retrieve into a string format suitable for the query
    col_string = ", ".join(col_to_retrieve)

    # Performing the query and retrieving the results in a pandas DataFrame
    job = tap.launch_job(query)
    results = job.get_results()
    # Displaying the columns of the `desi_dr1.photometry` table to check that the required fields for the magnitudes data are present
    print("\n[INFO] Columns of the `desi_dr1.photometry` table that will be downloaded:")
    for i, col in enumerate(results["column_name"]):
        if col in col_to_retrieve:
            print(f"\t{i}: {col}, {results["datatype"][i]}, {results["description"][i]}")
    print("\n")

    # Creation of Sparcl Client Instance
    client = SparclClient(read_timeout=60, announcement=False)
    # List of desired output fields
    outfields = ["sparcl_id", "ra", "dec", "redshift", "spectype", "data_release"]
    # Constraints on spectra to retrieve
    constraints = {
        "spectype"      :   ["QSO"],
        "redshift"      :   REDSHIFT_RANGE,
        "data_release"  :   ["DESI-DR1"],
    }

    # Initializing the list of DataFrames to concatenate for the final results
    results_dfs = []

    # Inform user
    print("\n[INFO] Retrieving magnitudes data from the DESI-DR1 photometric database...")

    # RA step (in degrees)
    ra_step = 1
    # Loop on RA slices
    for ra_start in tqdm(range(0, 360, ra_step), desc="Scanning RA", unit="slice"):

        # Defining the RA end for the current slice
        ra_end = ra_start + ra_step
        # Updating the RA constraint for the current slice
        constraints["ra"] = [ra_start, ra_end]

        # Query execution
        found = client.find(constraints=constraints, outfields=outfields, limit=NUMBER_OF_SPECTRA)
        # Fields to retrieve using the IDs found
        fields = ["targetid"]
        # Retrieve spectra data using the IDs found
        retrieved = client.retrieve(uuid_list=found.ids, include=fields, dataset_list=["DESI-DR1"], limit=NUMBER_OF_SPECTRA)
        # Converting target IDs to a string format suitable for the query
        targetids = [record.targetid for record in retrieved.records]
        id_string = f"({','.join(map(str, targetids))})"

        # If there are no target IDs found for the current RA slice, we skip the query and move to the next slice
        if len(targetids) == 0:
            tqdm.write(f"[INFO] No target IDs found for RA slice {ra_start} - {ra_end}. Skipping this slice.")
            continue

        # Creation of the query
        query = f"""
        SELECT {col_string}
        FROM desi_dr1.photometry
        WHERE targetid IN {id_string}
        """
        # Performing the query and retrieving the results in a pandas DataFrame
        job = tap.launch_job(query)
        result = job.get_results()
        result_df = result.to_pandas()
        # Appending the results DataFrame to the list of results DataFrames to concatenate at the end
        results_dfs.append(result_df)
    
    # Concatenating the results DataFrames for each RA slice into a single DataFrame
    mag_table = pd.concat(results_dfs, ignore_index=True)

    # Renaming the columns of the magnitudes table to more human-readable names using the `MagColNames` enum
    mag_table = mag_table.rename(columns={
        "ra"                       :   MagColNames.RA,
        "dec"                      :   MagColNames.DEC,
        "flux_g"                   :   MagColNames.G_FLUX,
        "flux_r"                   :   MagColNames.R_FLUX,
        "flux_z"                   :   MagColNames.Z_FLUX,
        "mw_transmission_g"        :   MagColNames.MW_TRANS_G,
        "mw_transmission_r"        :   MagColNames.MW_TRANS_R,
        "mw_transmission_z"        :   MagColNames.MW_TRANS_Z,
        "flux_w1"                  :   MagColNames.W1_FLUX,
        "flux_w2"                  :   MagColNames.W2_FLUX,
        "flux_w3"                  :   MagColNames.W3_FLUX,
        "flux_w4"                  :   MagColNames.W4_FLUX,
        "mw_transmission_w1"       :   MagColNames.MW_TRANS_W1,
        "mw_transmission_w2"       :   MagColNames.MW_TRANS_W2,
        "mw_transmission_w3"       :   MagColNames.MW_TRANS_W3,
        "mw_transmission_w4"       :   MagColNames.MW_TRANS_W4,
        "gaia_phot_bp_mean_mag"    :   MagColNames.GAIA_BP_MAG,
        "gaia_phot_g_mean_mag"     :   MagColNames.GAIA_G_MAG,
        "gaia_phot_rp_mean_mag"    :   MagColNames.GAIA_RP_MAG,
        "targetid"                 :   MagColNames.TARGETID,
    })

    # Adding a "Name" column to the magnitudes table using the `generate_spectrum_name` function
    mag_table[MagColNames.NAME] = mag_table.apply(lambda row: generate_spectrum_name(row[MagColNames.RA], row[MagColNames.DEC]), axis=1)

    # Ordering the columns of the magnitudes table according to the `MagColNames` enum
    mag_table = mag_table[[col for col in MagColNames]]

    # Saving the magnitudes table in a local file in CSV format
    outputfile = f"{MAGNITUDES_DATA_FOLDER}mag_table.csv"
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    mag_table.to_csv(outputfile, index=False)

    # Returning to the main program
    return
