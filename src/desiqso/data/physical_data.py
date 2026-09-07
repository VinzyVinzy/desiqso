"""
This module contains the function to download the physical data associated with the DESI-DR1 
program for the quasar spectra processed, from the NoirLab database using TAP queries and 
the `SparclClient()` class.
"""

# Package imports
from astroquery.utils.tap.core import TapPlus
import os
import pandas as pd
from sparcl.client import SparclClient
from tqdm import tqdm

# Local imports
from src.desiqso.config import (PHYSICAL_DATA_FOLDER, REDSHIFT_RANGE, NUMBER_OF_SPECTRA,)
from src.desiqso.constants import PhysColNames
from src.desiqso.data.spectra_utils import generate_spectrum_name

# Function to download physical data from the DESI-DR1 database using TAP queries
def download_physical_data() -> None:
    """
    This function downloads the physical data  associated with the DESI-DR1 program for 
    the quasar spectra processed, from the NoirLab database using TAP queries and the 
    `SparclClient()` class. The physical data includes information such as the target ID, 
    right ascension, declination, morphology type, shape parameters, and Sersic index for 
    each quasar spectrum. The function retrieves the data in slices of right ascension 
    (RA) to manage the query size and avoid timeouts. The retrieved data is then saved in 
    a CSV file in the specified physical data folder.

    :return: None
    """

    # Inform user
    print("\n[INFO] Downloading physical data from the NoirLab database...\n")

    # Creation of a TapPlus instance to perform TAP queries on the DESI-DR1 database
    tap = TapPlus(url="https://datalab.noirlab.edu/tap")

    # Configuration of the base TAP query 
    query = """
    SELECT column_name, datatype, description
    FROM tap_schema.columns
    WHERE table_name = 'desi_dr1.photometry'
    """

    # List of columns to retrieve from the desi_dr1.photometry table
    col_to_retrieve = ["targetid", "dec", "ra", "morphtype", "dchisq_psf", "dchisq_rex", "dchisq_dev", "dchisq_exp", "dchisq_ser", "shape_r", "shape_e1", "shape_e2", "sersic"]
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
    data = pd.concat(results_dfs, ignore_index=True)

    # Renaming the columns of the physical data table to more human-readable names using the `PhysColNames` enum
    data = data.rename(columns={
        "targetid"  :   PhysColNames.TARGETID,
        "ra"        :   PhysColNames.RA,
        "dec"       :   PhysColNames.DEC,
        "morphtype" :   PhysColNames.MORPH_TYPE,
        "dchisq_psf":   PhysColNames.DCHISQ_PSF,
        "dchisq_rex":   PhysColNames.DCHISQ_REX,
        "dchisq_dev":   PhysColNames.DCHISQ_DEV,
        "dchisq_exp":   PhysColNames.DCHISQ_EXP,
        "dchisq_ser":   PhysColNames.DCHISQ_SER,
        "shape_r"   :   PhysColNames.SHAPE_R,
        "shape_e1"  :   PhysColNames.SHAPE_E1,
        "shape_e2"  :   PhysColNames.SHAPE_E2,
        "sersic"    :   PhysColNames.SERSIC,
    })

    # Adding a "Name" column to the physical data table using the `generate_spectrum_name` function
    data[PhysColNames.NAME] = data.apply(lambda row: generate_spectrum_name(row[PhysColNames.RA], row[PhysColNames.DEC]), axis=1)

    # Ordering the columns of the physical data table according to the `PhysColNames` enum
    data = data[[col for col in PhysColNames]]

    # Saving the physical data table in a local file in CSV format
    outputfile = f"{PHYSICAL_DATA_FOLDER}physical_data.csv"
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    data.to_csv(outputfile, index=False)

    # Returning to the main program
    return
