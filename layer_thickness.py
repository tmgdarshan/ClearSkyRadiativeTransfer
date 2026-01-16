"""
Calculate the clear sky upwelling longwave radiance at the top of the atmosphere. The atmosphere
is a standard tropical atmosphere, containing only water vapor, carbon dioxide and ozone
as trace gases. The water vapor absorption continuum is included.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import typhon
import pyarts3 as pa
import single_column_atmosphere as sca

pa.data.download()

MIXING_RATIO_CO2 = 400e-6
MIXING_RATIO_O3 = 1e-6
T_SURF = 290  # K

KAYSER_GRID = np.linspace(1, 2000, 200)
FREQ_GRID = pa.arts.convert.kaycm2freq(KAYSER_GRID)


def set_up_atmosphere(temp_profile, pressure_profile, H2O_profile, CO2_concentration):
    """
    Convert vertical profiles to xarray.Dataset for ARTS.

    Parameters
    ----------
    temp_profile : ndarray
        Temperature profile [K]
    pressure_profile : ndarray
        Pressure levels [Pa]
    H2O_profile : ndarray
        Water vapor VMR [mol/mol]
    CO2_concentration : float
        CO2 VMR [mol/mol]

    Returns
    -------
    xarray.Dataset
        Atmospheric state with T, p, H2O, O3, CO2 profiles
    """
    heights = typhon.physics.pressure2height(pressure_profile)

    atm = xr.Dataset(
        {
            "t": ("alt", temp_profile),
            "p": ("alt", pressure_profile),
            "H2O": ("alt", H2O_profile),
            "O3": ("alt", np.ones_like(pressure_profile) * MIXING_RATIO_O3),
            "CO2": ("alt", np.ones_like(pressure_profile) * CO2_concentration),
        },
        coords={"alt": heights, "lat": 0, "lon": 0},
    )

    return atm


def main():
    """
    Generate atmosphere and compute layer thicknesses.
    """
    # Generate profiles
    t_profile, wmr_profile, pressure_levels = sca.create_vertical_profile(T_SURF)
    atmosphere = set_up_atmosphere(t_profile, pressure_levels, wmr_profile, MIXING_RATIO_CO2)

    # Compute layer thicknesses
    heights = typhon.physics.pressure2height(pressure_levels)
    print("height shape:", np.shape(heights))
    print("heights range:", heights[0], "to", heights[-1], "m")

    dz = np.diff(heights, prepend=heights[0])
    print("dz shape:", np.shape(dz))
    print("dz range:", dz[0], "to", dz[-1], "m")

    h = np.cumsum(dz)
    print("h shape:", np.shape(h))
    print("h range:", h[0], "to", h[-1], "m")


if __name__ == "__main__":
    main()
