""""""

# Packages import
import os
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Local imports
from src.desiqso.analysis.absorption_masks import compute_h2_absorption_masks
from src.desiqso.analysis.cross_correlation import (NUMBER_OF_CORES, select_spectra_for_analysis, spectrum_analysis, init_worker,)
from src.desiqso.config import MOCK_ANALYSIS_FOLDER
from src.desiqso.constants import (ColNames, Modes, Categories,)
from src.desiqso.data.loader import load_spectrum_from_filename
from src.desiqso.models.dataset import AnalysisResults
from src.desiqso.models.profile import ProfileManager
from src.desiqso.models.results import CrossCorrelationResult
from src.desiqso.models.spectrum import SpectrumRecord

# Function to retrieve and save the list of spectra files with high SNR
def mock_analysis() -> dict[str, SpectrumRecord]:
    """"""

    # Retrieve test sample
    sample_spectra = select_spectra_for_analysis(mode=Modes.SAMPLE)
    
    # Loading the cross-correlation analysis results
    AnalysisResults.load_results(verbose=False)
    # Retrieving the table corresponding to the spectra with high SNR and with low chances of having H₂
    table = AnalysisResults.results_survey(mode=Modes.ALL, profile_name="best", thresholds_dict={ColNames.SNR : (15, None), ColNames.CORR_PARAM : (None, 0.3)})
    # Selecting only spectra that are not from the "confirmed", "borderline" or "rejected" categories
    table = table[table[ColNames.CATEGORY] == Categories.OTHER]
    # Selecting spectra that are not already in the sample
    table = table[~table[ColNames.FILENAME].isin(sample_spectra)]

    # Keeping only a sample of a 1000 spectra filenames with a fixed seed for reproducibility
    filenames = table[ColNames.FILENAME].sample(n=1000, random_state=0).tolist()
    # Adding the other spectra to the sample
    sample_spectra.extend(table[~table[ColNames.FILENAME].isin(filenames)][ColNames.FILENAME].tolist())

    # Loading synthetic H₂ profiles and retrieving the first one
    ProfileManager.load_all(verbose=False)
    profile = ProfileManager.all_profiles()[0]

    # Initialize dictionary containing the filenames and corresponding records of the mock spectra
    mock_spectra = {}

    # Looping on the filenames
    for filename in filenames:

        # Opening the `.fits` file corresponding as a `SpectrumRecord` instance
        record = load_spectrum_from_filename(filename)
        # Rebinning the synthetic profile to match the record
        _, _, flux_rebinned = compute_h2_absorption_masks(record.wavelength, record.redshift, profile, record.mask)
        # Adding the synthetic profile to the original spectrum
        record.flux *= flux_rebinned
        # Adding guassian noise to the mock spectrum
        snr = np.random.randint(3, 10)
        noise = np.random.normal(0, 1/snr, len(record.wavelength))
        record.flux += noise
        # Adding the record to the sample list
        sample_spectra.append(filename)
        # Adding the record to the dictionary
        mock_spectra[filename] = record
    
    # Creation of the results accumulator for the spectra analysis, to store the results of the cross-correlation analysis for all the loaded spectra
    results_accumulator = defaultdict(list)
    
    # Inform user
    print(f"\n[INFO] Processing spectra in parallel, using {NUMBER_OF_CORES} cores...")

    # Creation of Process Pool Executor for parallel processing
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), NUMBER_OF_CORES), initializer=init_worker) as executor:
        # Parallel processing of the cross-correlation analysis using the Process Pool Executor
        futures = [executor.submit(spectrum_analysis, spectrum_file) for spectrum_file in sample_spectra]
        # Loop on the futures to retrieve the results of the cross-correlation analysis
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing spectra", unit="file"):
            # Retrieve the results of the cross-correlation analysis
            result = future.result()
            # If the cross-correlation analysis failed
            if result is None:
                continue
            # If the cross-correlation analysis succeeded, retrieve the file name and the results
            file_name, results = result
            # Update the results accumulator using the file name and the results
            results_accumulator[file_name] = results

    # Inform user
    print(f"\n[INFO] Spectra processing completed successfully! {len(results_accumulator)} spectra processed in total.")
    
    # ================
    # Results saving
    # ================

    # Inform user
    print("\n[INFO] Saving spectra analysis results...")
    # Initialisation of results files
    if CrossCorrelationResult._results is None:
        CrossCorrelationResult.initialize_results(output_folder=MOCK_ANALYSIS_FOLDER)
    # Save the results of the spectra analysis in the corresponding local files in .txt format
    for _, results in tqdm(results_accumulator.items(), desc="Saving results", unit="file"):
        results.save_results()
    # Inform user
    print(f"\n[INFO] Spectra analysis results saved successfully in folder `{MOCK_ANALYSIS_FOLDER}`!\n")

    # Return the list of mock spectra filenames
    return mock_spectra