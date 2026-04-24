"""
This module contains functions to generate synthetic H₂ profiles using the VoigtFit package.
"""

# Do not show numpy warnings (neccessary for the VoigtFit package)
import warnings
warnings.filterwarnings("ignore",
    message="Input line 1 contained no data.*",
    category=UserWarning
)

# Packages import
from astropy.constants import c
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from VoigtFit.funcs.voigt import Voigt, convolve_profile
from VoigtFit.container.lines import lineList
from VoigtFit.utils.molecules import H2, energy_of_level

# Local imports
from src.desiqso.config import SYNTHETIC_PROFILES_FOLDER, FIGURES_FOLDER, settings
from src.desiqso.constants import C_KMS


# Function to synthetize an H₂ profile from a list of bands and a few parameters
def generate_h2_profile(resolution_power : float = 2000., Ntot : float = 1e20, T_exc : list[float] = [100., 100.,100.,100.,100.,100.,100.,], Jmax : int = 1, b_param : float = 5., pixel_size : float = 5., save : bool = False, verbose : bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    This function generates a synthetic H₂ profile from a list of bands and a few parameters. The
    bands currently supported are: BX(0-0), BX(1-0), BX(2-0), BX(3-0), BX(4-0), BX(5-0)), BX(6-0), 
    BX(7-0), BX(8-0), BX(9-0)), CX(0-0), CX(1-0), CX(2-0), CX(3-0), CX(4-0), CX(5-0)).

    :param resolution_power: The resolution power of the synthetic profile, corresponding to the 
    instrument. Default is 2000.
    :type resolution_power: float, optional
    :param Ntot: The total column density of H₂ in the synthetic profile. Default is 1e20.
    :type Ntot: float, optional
    :param T_exc: The excitation temperatures of the the rotational states of the synthetic profile. Default is 100 for every level.
    :type T_exc: list[float], optional
    :param Jmax: The maximum rotational level of the synthetic profile. Default is 1.
    :type Jmax: int, optional
    :param b_param: The velocity width of the Voigt profile (Doppler parameter). Default is 5.
    :type b_param: float, optional
    :param pixel_size: The pixel size of the synthetic profile. Default is 5.
    :param save: Flag to save the synthetic profile. Default is False.    
    :type save: bool, optional
    :param verbose: Flag to print information about the synthetic profile. Default is True.
    :type verbose: bool, optional
    :return: A tuple containing the wavelength array and the flux array of the synthetic profile.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    # ================
    # Inform user and basic configuration
    # ================

    # Inform user of selected parameters
    if verbose:
        print(f"\n[INFO] Generating synthetic profile with the following configuration:\n")
        print(f"[INFO] - 2 rotational levels (J=0,1)")
        print(f"[INFO] - 10 bands (BX(0-0), BX(1-0), BX(2-0), BX(3-0), BX(4-0), BX(5-0)), BX(6-0), BX(7-0), BX(8-0), BX(9-0))")
        print(f"[INFO] - 6 bands (CX(0-0), CX(1-0), CX(2-0), CX(3-0), CX(4-0), CX(5-0))")
        print(f"[INFO] - Resolution power : {resolution_power}")
        print(f"[INFO] - Constant pixel size in km/s : {pixel_size} km/s")
        print(f"[INFO] - Excitation temperature of ground state : {T_exc[0]} K")
        print(f"[INFO] - Velocity width of the Voigt profile (Doppler parameter) : {b_param} km/s\n")

    # Update and apply matplotlib settings
    settings["xtick.top"] = True
    plt.rcParams.update(**settings)
    # Turn off interactive mode to solve a display bug
    plt.ioff()

    # ================
    # Configuration for the synthetic H₂ profile
    # ================

    # Number of rotational levels to add to the synthetic profile
    NUM_ROTATIONAL_LEVELS = min(Jmax + 1, 7) # (2 => J=0,1 ;  4 => J=0,1,2,3)

    # List of redshifts for each rotational level (defaults to 0, but can be modified if needed)
    redshift = [0. for _ in range(NUM_ROTATIONAL_LEVELS)]

    # Velocity width of the Voigt profile (cm/s) (also called Doppler parameter)
    velocity_width = b_param * 1e5 # to convert in km/s

    # List of bands to add to synthetic profile
    # BX(v'-v) : electronic transition from rotational level v' to v
    bands_to_add = ["BX(0-0)", "BX(1-0)", "BX(2-0)", "BX(3-0)", "BX(4-0)", "BX(5-0)", "BX(6-0)", "BX(7-0)", "BX(8-0)", "BX(9-0)",
                    "CX(0-0)", "CX(1-0)", "CX(2-0)", "CX(3-0)", "CX(4-0)", "CX(5-0)",]

    # ================
    # Compute column density using a the J=0 rotational level column density and an excitation temperature
    # ================

    # Function to compute statistical weight for H₂ rotational level
    def g_J(J : float) -> float:
        # If J is even, return 2J+1
        if J % 2 == 0:
            return (2. * J + 1.)
        # If J is odd, return 3(2J+1)
        else:
            return 3 * (2. * J + 1.)
        
    # Initialize partition function Z
    Z=0.
    # Loop on rotational levels
    for J in range(NUM_ROTATIONAL_LEVELS):
        # Compute partition function Z
        Z += g_J(J) * math.exp(-(energy_of_level("H2",J)-energy_of_level("H2",0))/T_exc[J])
        
    # Initialize an empty list for column densities
    column_density : list[float] = []

    # Loop on rotational levels
    for J in range(NUM_ROTATIONAL_LEVELS):
        # Compute column density of rotational level J
        column_density.append(Ntot * (g_J(J) * math.exp(-(energy_of_level("H2",J)-energy_of_level("H2",0))/T_exc[J])) / Z)

    # ================
    # Lines selection
    # ================

    # Inform user
    if verbose:
        print(f"[INFO] Selecting lines to add to the synthetic profile...\n")

    # Initialize an empty list for selected lines to add
    selected_lines = []
    # Initialize an empty list for rotational levels associated to the selected lines
    rotational_levels = []
    # Loop on bands to add
    for band in bands_to_add:
        # Loop on rotational levels (here, J=0,1,2)
        for J in range(NUM_ROTATIONAL_LEVELS):
            # Add lines to the list
            selected_lines.extend(H2[band][J])
            # Add rotational levels
            rotational_levels.extend([J]*len(H2[band][J]))

    # ================
    # Lines properties retrieval
    # ================

    # Inform user
    if verbose:
        print(f"[INFO] Retrieving lines properties...\n")

    # Load all available lines data from the `VoigtFit` package in a dictionary for easier retrieval of the selected lines properties
    lines_dict = {line[0] : line for line in lineList}

    # Initialize lists for selected lines properties
    lambda0             = []
    oscillator_strength = []
    gamma               = []

    # Retrieve the properties of the selected lines
    for line_name in selected_lines:
        # If the line is in the selected lines list
        line = lines_dict[line_name]
        # Add its properties to the lists
        lambda0.append(line[2])
        oscillator_strength.append(line[3])
        gamma.append(line[4])

    # Converting lists to arrays
    lambda0             = np.array(lambda0, dtype=float)
    oscillator_strength = np.array(oscillator_strength, dtype=float)
    gamma               = np.array(gamma, dtype=float)

    # ================
    # Profile generation
    # ================

    # Inform user
    if verbose:
        print(f"[INFO] Generating synthetic profile...\n")

    # Initalizing lambda0 and optical depth arrays
    wavelength = wl_grid_const_speed(800, 1400, pixel_size)
    tau = np.zeros_like(wavelength)

    # Loop on selected lines properties
    for (l0, f, gam, J) in zip(lambda0, oscillator_strength, gamma, rotational_levels):
        # Compute the optical depth for each line using the Voigt profile
        tau += Voigt(wavelength, l0, f, N=column_density[J], b=velocity_width, gam=gam, z=redshift[J])

    # Compute the profile transmission using the Beer-Lambert law
    profile = np.exp(-tau)

    # Compute the Line Spreading Function (LSF) sigma using the Full-Width at Half-Maximum 
    # of the synthetic profile (computed with the input resolution power and the pixel size)
    fwhm = wavelength / resolution_power 
    lsf_sigma = np.mean((fwhm / 2.35482) / (wavelength * (pixel_size / C_KMS)))  # wavelength simplifies in this equation to obtain a constant LSF sigma

    # Convolve with the instrument resolution
    convolved_profile = convolve_profile(profile, lsf_sigma)

    # ================
    # Plotting synthetic profile
    # ================

    # Inform user
    if verbose:
        print(f"[INFO] Plotting synthetic profile...\n")

    # Initializing the plot
    _, ax = plt.subplots(figsize=(12,8))

    # Plotting the transmission profile
    plt.plot(wavelength, convolved_profile)
    plt.xlabel(r"Wavelength [$\AA$]")
    plt.ylabel("Transmission")

    # Plot title
    plt.title(f"Synthetic "+r"H$_2$"+rf" profile with: R={resolution_power}, log(N$_{{tot}}$)={math.log10(Ntot):.1f} "+r"cm$^{-2}$"+f" and T={T_exc[0]} K")

    # Displaying the column density of each rotational level
    column_densities_str = ""
    for J in range(NUM_ROTATIONAL_LEVELS):
        column_densities_str += f"N(J={J}): {column_density[J] :.2e} "+r"cm$^{-2}$"+"\n"

    ax.text(0.8, 0.18, column_densities_str, transform=ax.transAxes,fontsize=12,va='top')

    # Setting the x-axis limits
    plt.xlim(875, 1175)
    # Setting the y-axis limits
    plt.ylim(0, 1.1)

    # ================
    # Saving the synthetic profile
    # ================

    # Saving the synthetic H₂ profile, if required
    if save:
        # Generating the synthetic H₂ profile name
        plot_name = f"h2_profile_res-{resolution_power:.1f}_ntot-{math.log10(Ntot):.1f}_J-{"-".join([str(J) for J in range(NUM_ROTATIONAL_LEVELS)])}_Texc-{T_exc[0]}_b-{b_param}_pix-{pixel_size}.png"
        # Creating the output folder
        output_path = f"{FIGURES_FOLDER}synthetic_profiles/{plot_name}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Saving the figure in the right folder
        plt.savefig(output_path, dpi=400)

        # Informing user
        print(f"[INFO] Synthetic H2 profile successfully saved in {SYNTHETIC_PROFILES_FOLDER}plots/{plot_name}\n")
    
    # Closing the plot
    plt.close()

    # Returning the wavelength and the convolved profile for further use if needed
    return wavelength, convolved_profile

# Function to generate a wavelength grid with a pixel size constant in speed
def wl_grid_const_speed(min_wl : float, max_wl : float, delta_v : float) -> np.ndarray:
    """
    Generate a wavelength grid with a pixel size constant in speed.

    :param min_wl: Minimum wavelength of the grid.
    :type min_wl: float
    :param max_wl: Maximum wavelength of the grid.
    :type max_wl: float
    :param delta_v: Pixel size in km/s.
    :type delta_v: float
    :return: Wavelength grid with the specified properties.
    :rtype: np.ndarray

    Notes
    -----
    The wavelength grid is generated using a logarithmic spacing, with the
    pixel size in speed being constant. This means that the wavelength
    increment between two consecutive pixels is proportional to the
    wavelength value. The grid is generated using the `np.logspace`
    function.

    Physical principle: dl/l = dv/c => dlog(l) = dv/c => v = c*dlog(l)
    """

    # Define the logarithmic spacing for the wavelength grid
    dwl_log = delta_v / C_KMS

    # Computing the number of pixels in the final grid
    n_pix = int(math.log10(max_wl / min_wl) / dwl_log) + 1

    # Generate the wavelength grid
    wl_grid = np.logspace(math.log10(min_wl), math.log10(max_wl), n_pix)

    # Return the wavelength grid
    return wl_grid
