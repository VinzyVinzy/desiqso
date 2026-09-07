"""
This module contains a function to download the residual images in 
the griz bands from the Legacy Survey Viewer website.
"""

# Packages import
import os
import pandas as pd
import requests
from tqdm import tqdm

# Local imports
from src.desiqso.config import (IMAGE_DATA_FOLDER,)
from src.desiqso.constants import (ColNames,)

# Function to download the residual images from the Legacy Survey Viewer website
def download_residual_images(table : pd.DataFrame) -> None:
    """
    This function downloads a cutout of the residual images in the griz bands from 
    the Legacy Survey Viewer website for each object in the input table, using its 
    RA and DEC coordinates. The images are saved in the IMAGE_DATA_FOLDER defined 
    in the `config.py` file.

    :param table: The table containing the objects for which the images are to be downloaded.
    :type table: pd.DataFrame
    :return: None
    """

    # If the table provided is empty
    if len(table) == 0:
        # Informing the user that there are no images to download
        print("\n[INFO] No images to download.")
        return

    # Informing the user that the images are being downloaded
    print("\n[INFO] Downloading images...\n")

    # Iterating over the rows of the table and downloading the images
    for _, row in tqdm(table.iterrows(), total=len(table), desc="Downloading images", unit="image"):

        # Defining the filename to save the image in
        filename = os.path.join(IMAGE_DATA_FOLDER, row[ColNames.FILENAME][:-5] + ".fits")

        # Defining the layer to download the images from
        layer = "ls-dr10-resid"
        # Defining the band to download the images from
        band = "griz"

        # Defining the URL to download the image from
        url = (f"https://www.legacysurvey.org/viewer/fits-cutout?ra={row[ColNames.RA]}&dec={row[ColNames.DEC]}&layer={layer}&pixscale=0.262&bands={band}&size=50")

        # Requesting the image
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Saving the image
        with open(filename, "wb") as file:
            # Writing the content of the response to the file
            file.write(response.content)

        # Displaying a message to the user
        tqdm.write(f"Image downloaded for {row[ColNames.NAME]}. Size of the image: {os.path.getsize(filename)/1e6:.2f} MB")
    
    # Informing the user that the images have been downloaded
    print("\n[INFO] Images downloaded successfully.\n")
        
    # Returning to the main function
    return
