"""
This module contains functions to find and save new candidates in a file.
"""

# Packages import
import os
from tqdm import tqdm

# Local import
from src.desiqso.config import NEW_CANDIDATES_PATH
from src.desiqso.constants import (ColNames, Modes,)
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.models.profile import ProfileManager

# Function to find and save new candidates in a file
def find_new_candidates() -> None :
    """
    This function finds and saves new candidates in a file.
    """
    
    # Inform user
    print("\n[INFO] Finding new candidates...\n") 

    # Loading cross-correlation analysis results
    AnalysisResults.load_results(verbose=False)
    # Loading all the synthetic profiles
    ProfileManager.load_all(verbose=False)

    # Loading visual inspection results in a single table
    visual_table = AnalysisResults._visual_inspection

    # Loading the results of the cross-correlation analysis
    table = AnalysisResults.results_survey(mode = Modes.VALID, profile_name="all", thresholds_dict={},)

    # Looping over the table to find new candidates
    for _, row in tqdm(table.iterrows(), total=len(table), desc=f"Saving new candidates", unit="spectra"):
        # If the candidate is not in the visual inspection results table
        if row[ColNames.FILENAME] not in visual_table[ColNames.FILENAME].values:
            # Saving the new candidate
            save_new_candidate(row[ColNames.FILENAME], row[ColNames.PROFILE], row[ColNames.GRADE])
    
    # Inform user
    print(f"\n[INFO] New candidates saved in file `{NEW_CANDIDATES_PATH}`.\n")

    # Return to the main programm
    return

# Function to save a new candidate name and profile index in a file
def save_new_candidate(candidate_filename : str, profile_name : str, grade : int) -> None :
    """
    Function to save a new candidate name and profile index in a file.

    It takes as input the name of the candidate and the associated synthetic profile DataFrame.
    It checks if the output folder exists and reads the existing file.
    If the candidate name is found in the existing file, it replaces the line with the new profile name.
    If the candidate name is not found, it adds a new line with the candidate name and profile name.
    It writes the new file and informs the user about the new saved candidate.

    :param candidate_name: The name of the candidate.
    :type candidate_name: str
    :param profile: The associated synthetic profile DataFrame.
    :type profile: pd.DataFrame
    :return: None
    :rtype: None
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(NEW_CANDIDATES_PATH), exist_ok=True)

    # Create a list to store the lines and a flag to check if the profile name was found
    lines = []
    found = False

    # Check if the output folder exists
    if os.path.exists(NEW_CANDIDATES_PATH):
        # Read the existing file
        with open(NEW_CANDIDATES_PATH, "r", encoding="utf-8") as file:
            # Loop on the lines
            for line in file:
                # Split the line to retrieve the profile name associated to the expected core transmission
                name, saved_profile_name, _ = line.strip().split("\t")
                # If the profile name and the candidate name are the same, replace the line
                if name == candidate_filename and saved_profile_name == profile_name:
                    # Replace the expected core transmission
                    lines.append(f"{candidate_filename}\t{profile_name}\t{grade}\n")
                    found = True
                # Else, keep the existing line
                else:
                    lines.append(line)
    
    # If the profile name was not found in the existing file
    if not found:
        # Add the new line
        lines.append(f"{candidate_filename}\t{profile_name}\t{grade}\n")

    # Write the new file
    with open(NEW_CANDIDATES_PATH, "w", encoding="utf-8") as file:
        file.writelines(lines)
    
    # Inform the user
    tqdm.write(f"New candidate: {candidate_filename} with profile {profile_name}. Saved in {NEW_CANDIDATES_PATH}.")

    # Return to the main program
    return
