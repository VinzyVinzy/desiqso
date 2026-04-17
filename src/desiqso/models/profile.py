"""
This module contains a class to store/manage synthetic profiles and a class to represent a synthetic profile.
Their respective methods and properties allow for easier access and manipulation of the synthetic profiles.
"""

# Importing necessary libraries
from dataclasses import dataclass
import math
import numpy as np
import os
from typing import Dict

# Local imports
from src.desiqso.config import SYNTHETIC_PROFILES_FOLDER, USE_BASIC_SYNTHETIC_PROFILES
from src.desiqso.data.synthetic_profiles import generate_h2_profile
from src.desiqso.utils.helpers import get_profile_characteristics

# Class containing all profiles for easier access
class ProfileManager:
    """
    This class stores all synthetic profiles for easier and centralized access.
    It has a class attribute `_profiles` which is a dictionary of `Profile` objects
    indexed by their filenames.

    It contains the following class methods:
    - `all_profiles()`: returns a list of all profiles from the class attribute `_profiles`.
    - `load_all()` : loads all synthetic profiles from local files into the class attribute `_profiles`.
    - `get(name:str)`: returns the profile with the given name from the class attribute `_profiles`.
    - `get_profile_index(name:str)`: returns the index corresponding to a synthetic profile using its name.
    """

    # Class attribute to store all synthetic profiles
    _profiles:Dict[str,"Profile"] = {}

    # Property to retrieve a list of all profiles from the class attribute `_profiles`
    @classmethod
    def all_profiles(cls) -> list["Profile"]:
        """
        This class method returns a list of all profiles from the class attribute `_profiles`.
        """

        # Returning the list of profiles
        return list(cls._profiles.values())

    # Class method to load all synthetic profiles
    @classmethod
    def load_all(cls, verbose: bool = True) -> None:
        """
        This class method loads all synthetic profiles from local files into the class 
        attribute `_profiles`, which is a dictionary of `Profile` objects.

        :param verbose: Flag specifying whether to print information about the loading process. Defaults to True.
        :type verbose: bool
        """

        # Inform user
        if verbose:
            print("\n[INFO] Loading synthetic profiles...")

        # Clear the dictionary to prevent duplicates
        cls._profiles.clear()

        # Condition to load synthetic H₂ profiles from local files from Pasquier
        if USE_BASIC_SYNTHETIC_PROFILES:

            # Load synthetic H₂ profile files from disk.
            # Each file contains two columns: wavelength and normalized flux.
            profile_data_a = np.loadtxt("data/external/synthetic_profiles/h2profile20.0.dat")
            profile_data_b = np.loadtxt("data/external/synthetic_profiles/h2profile21.0.dat")

            # Extract wavelength and flux columns for profile A
            wavelength_template_a = profile_data_a[:, 0]
            flux_template_a = profile_data_a[:, 1]

            # Extract wavelength and flux columns for profile B
            wavelength_template_b = profile_data_b[:, 0]
            flux_template_b = profile_data_b[:, 1]

            # Add the data to the list
            cls._profiles["synth_a"] = Profile(wavelength_template_a, flux_template_a, T_exc=100.0, Jmax=1, N0=1e20, b_param=5., resolution_power=2000., pixel_size=5.)
            cls._profiles["synth_b"] = Profile(wavelength_template_b, flux_template_b, T_exc=100.0, Jmax=1, N0=1e21, b_param=5., resolution_power=2000., pixel_size=5.)

        # If there are also more profiles, add them here
        if os.path.exists(SYNTHETIC_PROFILES_FOLDER):
            # Loop on all files in the folder
            for file in os.listdir(SYNTHETIC_PROFILES_FOLDER):
                # Load the data from the .npy file
                profile_data = np.load(SYNTHETIC_PROFILES_FOLDER+file)
                # Extract the wavelength and flux columns
                wavelength_template = profile_data[:, 0]
                flux_template = profile_data[:, 1]
                # Extract the profile characteristics from its filename
                T_exc, J, N_0, res, b_param, pix_size = get_profile_characteristics(file)
                # Add the profile to the dictionary, using the filename as the key
                cls._profiles[file[:-4]] = Profile(wavelength_template, flux_template, T_exc=float(T_exc), Jmax=float(J[-1]), N0=10**float(N_0), b_param=float(b_param), resolution_power=float(res), pixel_size=float(pix_size))
        
        # Inform user
        if verbose:
            print(f"[INFO] All {len(cls._profiles)} synthetic profile(s) loaded.\n")

        # Return to the main function
        return

    # Class method to retrieve a profile from the class attribute `_profiles`
    @classmethod
    def get(cls, name:str) -> "Profile":
        """
        This class method returns the profile with the given name from the class attribute `_profiles`.
        """

        # Checking if the profile exists
        if name not in cls._profiles:
            # If not, raising an error
            raise KeyError(f"Profile `{name}` not found in the loaded profiles.")
        # Returning the profile retrieved
        return cls._profiles[name]

    # Class method to retrieve the index corresponding to a synthetic profile using its name
    @classmethod
    def get_profile_index(cls, name:str) -> int:
        """
        This class method returns the index corresponding to a synthetic profile using its name.
        """

        # Returning the index
        return list(cls._profiles.keys()).index(name)
    

# Class representing a profile for easier handling of the synthetic H₂ profiles data for the spectra analysis
@dataclass
class Profile:
    """
    Class representing a profile for easier handling of the synthetic H₂ profiles data for the spectra analysis.
    
    It contains all the characteristics of a synthetic H₂ profile, which are stored as class attributes. Currently,
    the class attributes are: `wavelength`, `flux`, `T_exc`, `Jmax`, `N0`, `b_param`, `resolution_power`, `pixel_size`.
    
    It also contains the `name` and `legend_label` properties, which are used in the plots and for profile identification.

    Finally, it contains the following class methods:
    - `from_synthetic`: factory method to create a `Profile` instance from a synthetic profile data for easier 
    conversion of the loaded synthetic profile data into a python object to manipulate it for the spectra analysis.
    - `save`: method to save the synthetic profile data to a file for later use in the spectra analysis.
    """

    # Class attributes representing the synthetic profile data
    wavelength: np.ndarray
    flux: np.ndarray
    T_exc: float
    Jmax : float
    N0: float
    b_param: float
    resolution_power: float
    pixel_size: float

    # Property representing the name of the synthetic profile
    @property
    def name(self) -> str:
        """
        Returns the standardized name of the synthetic profile.
        """

        # Returning the name
        return f"h2_profile_res-{self.resolution_power:.1f}_n0-{math.log10(self.N0):.1f}_J-{'-'.join([str(J) for J in range(int(self.Jmax)+1)])}_Texc-{self.T_exc}_b-{self.b_param}_pix-{self.pixel_size}"
    
    # Property representing the legend label of the synthetic profile, used in the plots
    @property
    def legend_label(self) -> str:
        """
        Returns the legend label of the synthetic profile, containing its characteristics.
        """

        # Creating the legend label using the profile characteristics
        label = (
            rf"Synthetic profile with "
            rf"$T_{{exc}} = {self.T_exc}\,\mathrm{{K}}, "
            rf"J = {'-'.join([str(J) for J in range(int(self.Jmax)+1)])}, "
            rf"N_0 = 10^{{{math.log10(self.N0)}}}\,\mathrm{{cm^{{-2}}}}, "
            rf"R = {self.resolution_power},"
            rf"b = {self.b_param}\,\mathrm{{km.s^{{-1}}}}$"
        )
        # Returning the legend label
        return label

    # Factory method to create a `Profile` instance from a synthetic profile data for easier conversion of the loaded synthetic profile data into a python object to manipulate it for the spectra analysis.
    @classmethod
    def from_synthetic(cls, resolution_power : float = 2000., N0 : float = 1e20, T_exc : float = 100., Jmax : int = 1, b_param : float = 5., pixel_size : float = 5., save : bool = True, verbose : bool = True) -> "Profile":
        """
        Factory method to create a `Profile` instance from a synthetic profile data for easier 
        conversion of the loaded synthetic profile data into a python object to manipulate it 
        for the spectra analysis.
        """

        # Generating the synthetic profile wavelength and flux using the provided parameters and the `generate_h2_profile` function
        wavelength, flux = generate_h2_profile(resolution_power=resolution_power, N0=N0, T_exc=T_exc, Jmax=Jmax, b_param=b_param, pixel_size=pixel_size, save=save, verbose=verbose)

        # Returning the `Profile` instance corresponding to the synthetic profile
        return cls(
            wavelength          = wavelength,
            flux                = flux,
            T_exc               = T_exc,
            Jmax                = Jmax,
            N0                  = N0,
            b_param             = b_param,
            resolution_power    = resolution_power,
            pixel_size          = pixel_size,
        )
    
    # Method to return a complete version of a synthetic profile
    def get_complete_profile(self) -> "Profile":
        """
        This method returns a complete version of a synthetic profile, which is generated using the same 
        parameters as the original profile but with a higher Jmax value (Jmax=7) to include more absorption 
        features in the synthetic profile. This complete version of the synthetic profile can be used for 
        visualization purposes to show all the absorption features that could be present in the spectrum, 
        even if they were not included in the cross-correlation analysis to determine the best fit redshift 
        and core transmission values.
        """

        # Returning the complete version of the synthetic profile
        return self.from_synthetic(resolution_power=self.resolution_power, N0=self.N0, T_exc=self.T_exc, Jmax=7, b_param=self.b_param, pixel_size=self.pixel_size, save=False, verbose=False)
    
    # Method to save the synthetic profile data to a file for later use in the spectra analysis
    def save(self) -> None:
        """
        This method saves the synthetic profile data to a file for later use in the spectra analysis.
        The folder where the synthetic profiles are saved is defined as the `SYNTHETIC_PROFILES_FOLDER` 
        constant in the `src.desiqso.config` module.
        """
        
        # Creating the SYNTHETIC_PROFILES folder, if it doesn't exist
        os.makedirs(f"{SYNTHETIC_PROFILES_FOLDER}", exist_ok=True)
        # Saving the synthetic H₂ profile in the `synthetic_profiles` folder
        profile_filename = f"{SYNTHETIC_PROFILES_FOLDER}{self.name}.npy"
        np.save(profile_filename, np.column_stack((self.wavelength, self.flux)))

        # Informing user
        print(f"\n[INFO] Synthetic H2 profile successfully saved in file `{profile_filename}`.\n")
    
    