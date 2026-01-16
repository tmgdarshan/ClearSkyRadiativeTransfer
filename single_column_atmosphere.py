"""
Single-column moist-adiabatic atmosphere.

Creates vertical profiles of temperature and water vapour volume mixing ratio
on a fixed pressure grid, using typhon's moist lapse rate formulation and a
simple cold-point tropopause at 200 K.

The profiles are intended for use in idealized clear-sky radiative transfer
experiments (e.g. with ARTS/PyARTS) and for basic diagnostics/visualisation.
"""

import matplotlib.pyplot as plt
import numpy as np
import typhon

# Surface and profile configuration
t_0 = 290                                  # K, surface temperature
COLD_POINT_TEMPERATURE_TROPOPAUSE = 200    # K, imposed tropopause temperature
RELATIVE_HUMIDITY = 0.8                    # -, between 0 and 1
NUMBER_OF_PRESSURE_LEVELS = 100

# Pressure grid in Pa: from 1000 hPa to 10 hPa
PRESSURE_LEVELS = np.logspace(3, 0, NUMBER_OF_PRESSURE_LEVELS) * 10**2  # Pa


def create_vertical_profile(surface_temperature: float):
    """
    Create a moist-adiabatic temperature and water vapour profile.

    The profile is computed on the global constant PRESSURE_LEVELS grid.
    Temperature is obtained by integrating the moist lapse rate from the
    surface upward, using typhon's implementation. Once the temperature
    reaches the specified cold-point value, the atmosphere is treated as
    an isothermal stratosphere with constant water vapour.

    Relative humidity is kept fixed at RELATIVE_HUMIDITY during the moist
    ascent, and the corresponding water vapour volume mixing ratio is
    diagnosed from pressure and temperature.

    Parameters
    ----------
    surface_temperature : float
        Surface temperature in Kelvin at the lowest pressure level.

    Returns
    -------
    t_profile : np.ndarray
        Temperature profile [K] on PRESSURE_LEVELS.
    water_vapor_volume_mixing_ratio : np.ndarray
        Water vapour volume mixing ratio [mol/mol] on PRESSURE_LEVELS.
    pressure_levels : np.ndarray
        Pressure grid [Pa] (identical to PRESSURE_LEVELS).

    Notes
    -----
    - The tropopause is treated diagnostically as the first level where
      T <= COLD_POINT_TEMPERATURE_TROPOPAUSE.
    - Above the tropopause, both temperature and water vapour are held
      constant (simple isothermal, well-mixed stratosphere assumption).
    """
    # create empty arrays to store data
    water_vapor_volume_mixing_ratio = np.zeros_like(PRESSURE_LEVELS)
    t_profile = np.zeros_like(PRESSURE_LEVELS)

    # pressure levels into height levels in m
    heights = typhon.physics.pressure2height(PRESSURE_LEVELS)

    # set first value to surface temperature
    t_profile[0] = surface_temperature  # t_0 is set to surface pressure to start from
    water_vapor_volume_mixing_ratio[0] = typhon.physics.relative_humidity2vmr(
        RELATIVE_HUMIDITY, PRESSURE_LEVELS[0], t_profile[0]
    )

    first_tp_p_lvl = True  # flag for first pressure level of tropopause

    # iterate through pressure levels
    for i in range(1, len(PRESSURE_LEVELS)):

        # Temperature gradient (moist lapse rate, K/m)
        lapse_rate = typhon.physics.moist_lapse_rate(
            PRESSURE_LEVELS[i - 1], t_profile[i - 1]
        )

        # Thickness of height levels
        delta_z = heights[i] - heights[i - 1]

        # new temperature using lapse rate
        t_profile[i] = t_profile[i - 1] - lapse_rate * delta_z

        # new wmr using new temperature
        water_vapor_volume_mixing_ratio[i] = typhon.physics.relative_humidity2vmr(
            RELATIVE_HUMIDITY, PRESSURE_LEVELS[i], t_profile[i]
        )

        # handle tropopause (store first cold-point level)
        if t_profile[i] <= COLD_POINT_TEMPERATURE_TROPOPAUSE and first_tp_p_lvl:
            pressure_level_tropopause = PRESSURE_LEVELS[i]
            water_vmr_tropopause = typhon.physics.relative_humidity2vmr(
                RELATIVE_HUMIDITY,
                pressure_level_tropopause,
                COLD_POINT_TEMPERATURE_TROPOPAUSE,
            )
            first_tp_p_lvl = False

        # atmosphere profile if stratosphere is reached
        if t_profile[i] <= COLD_POINT_TEMPERATURE_TROPOPAUSE:
            # keep constant stratospheric temperature and wmr
            t_profile[i] = COLD_POINT_TEMPERATURE_TROPOPAUSE
            water_vapor_volume_mixing_ratio[i] = water_vmr_tropopause

    return t_profile, water_vapor_volume_mixing_ratio, PRESSURE_LEVELS


def plot_profiles(t_profile, water_vapor_volume_mixing_ratio):
    """
    Plot temperature, pressure–height, and water vapour profiles.

    Creates a three-panel figure and saves it as ``vertical_profiles.pdf``:
    - Temperature vs pressure
    - Pressure vs height
    - Water vapour VMR vs pressure

    Parameters
    ----------
    t_profile : np.ndarray
        Temperature profile [K] on PRESSURE_LEVELS.
    water_vapor_volume_mixing_ratio : np.ndarray
        Water vapour volume mixing ratio [mol/mol] on PRESSURE_LEVELS.
    """
    heights = typhon.physics.pressure2height(PRESSURE_LEVELS) / 1000  # km
    pressures_hpa = PRESSURE_LEVELS / 100                             # Pa → hPa

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # ---- Temperature Profile ----
    ax = axes[0]
    ax.plot(t_profile, pressures_hpa)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title("Temperature")
    ax.invert_yaxis()
    ax.grid(True)

    # ---- Pressure vs Height ----
    ax = axes[1]
    ax.plot(pressures_hpa, heights)
    ax.set_xlabel('Pressure (hPa)')
    ax.set_ylabel('Height (km)')
    ax.set_title("Pressure profile")
    ax.grid(True)

    # ---- Water Vapor Mixing Ratio ----
    ax = axes[2]
    ax.plot(water_vapor_volume_mixing_ratio, pressures_hpa)
    ax.set_xlabel('Mixing Ratio (mol/mol)')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title("Water Vapor VMR")
    ax.invert_yaxis()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("vertical_profiles.pdf")
    plt.show()


def main():
    """
    Driver function for standalone execution.

    Computes a moist-adiabatic profile using ``t_0`` as surface temperature
    and produces the diagnostic plots.
    """
    vertical_profile = create_vertical_profile(t_0)
    temp_profile = vertical_profile[0]
    wmr_profile = vertical_profile[1]

    # plot temp, wmr and pressure profile
    plot_profiles(temp_profile, wmr_profile)


if __name__ == "__main__":
    main()
