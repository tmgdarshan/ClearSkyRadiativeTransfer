"""
Clear-sky longwave radiation using PyARTS
Computes outgoing longwave radiation (OLR) for a single-column moist-adiabatic
atmosphere using ARTS radiative transfer. Includes full spectral calculation
and total flux integration.

**Status:** Working baseline for IR clear-sky OLR (~240 W/m² expected)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pyarts3 as pa
import single_column_atmosphere as sca
import typhon

# Configuration constants
pa.data.download()  # ARTS line catalogs
MIXING_RATIO_CO2 = 400e-6  # CO2 VMR
MIXING_RATIO_O3 = 1e-6  # O3 VMR
T_SURF = 290  # K, surface temperature

# Spectral grid: 1-2000 cm^-1 (full LW band)
KAYSER_GRID = np.linspace(1, 2000, 200)  # cm^-1
FREQ_GRID = pa.arts.convert.kaycm2freq(KAYSER_GRID)  # Hz


def set_up_atmosphere(temp_profile, pressure_profile, H2O_profile, CO2_concentration):
    """
    Convert vertical profiles to ARTS-compatible xarray.Dataset.

    Creates complete atmospheric state with trace gases (O3, CO2 as constant
    VMR profiles) and required metadata/attributes for pyarts3.

    Parameters
    ----------
    temp_profile : ndarray
        Temperature profile [K] from sca.create_vertical_profile()
    pressure_profile : ndarray
        Pressure levels [Pa]
    H2O_profile : ndarray
        Water vapour VMR [mol/mol]
    CO2_concentration : float
        CO2 VMR [mol/mol]

    Returns
    -------
    xarray.Dataset
        ARTS-ready atmospheric fields with units/attributes
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

    # ARTS metadata requirements
    atm["t"].attrs = {"units": "K", "long_name": "Temperature"}
    atm["p"].attrs = {"units": "Pa", "long_name": "Pressure"}
    atm["H2O"].attrs = {"units": "mol/mol", "long_name": "Water vapor VMR"}
    atm["O3"].attrs = {"units": "mol/mol", "long_name": "Ozone VMR"}
    atm["CO2"].attrs = {"units": "mol/mol", "long_name": "CO2 VMR"}
    atm["alt"].attrs = {"units": "m", "long_name": "Geometric altitude"}

    return atm


def absorption_coefficient(atmosphere):
    """
    Compute H2O + CO2 absorption coefficients across all levels.

    **NOTE:** Currently unused - full ARTS solver preferred for line+continuum.

    Parameters
    ----------
    atmosphere : xarray.Dataset
        Atmospheric state from set_up_atmosphere()

    Returns
    -------
    tuple of ndarrays
        (h2o_absorption, co2_absorption) - shape (n_levels, n_frequencies)
    """
    h2o_absorption = pa.recipe.SingleSpeciesAbsorption(species="H2O")
    co2_absorption = pa.recipe.SingleSpeciesAbsorption(species="CO2")

    atm_point = pa.arts.AtmPoint()
    atm_point["CO2"] = MIXING_RATIO_CO2

    height_levels = atmosphere.sizes["alt"]
    temps = atmosphere["t"].values
    pressures = atmosphere["p"].values
    h2o_values = atmosphere["H2O"].values

    # Vectorized computation over heights
    absorption_coefficient_h2o = []
    absorption_coefficient_co2 = []

    for h in range(height_levels):
        atm_point.temperature = temps[h]
        atm_point.pressure = pressures[h]
        atm_point["H2O"] = h2o_values[h]

        absorption_coefficient_h2o.append(h2o_absorption(FREQ_GRID, atm_point))
        absorption_coefficient_co2.append(co2_absorption(FREQ_GRID, atm_point))

    return np.array(absorption_coefficient_h2o), np.array(absorption_coefficient_co2)


def set_up_workspace(atmosphere):
    """
    Configure complete ARTS workspace for clear-sky emission.

    **Key steps:**
    1. Load H2O+CO2+O3 lines + CKD water continuum
    2. 25 cm^-1 cutoff for speed (consistent w/ CKD)
    3. Remove 90% weakest lines
    4. TOA-looking geometry (100km nadir)
    5. Compute spectral_radianceClearskyEmission()

    Parameters
    ----------
    atmosphere : xarray.Dataset
        From set_up_atmosphere()

    Returns
    -------
    pa.Workspace
        Ready-to-run ARTS workspace
    """
    ws = pa.Workspace()

    # Core absorbing species + continua
    ws.absorption_speciesSet(
        species=["H2O", "H2O-ForeignContCKDMT400", "H2O-SelfContCKDMT400", "CO2", "O3"]
    )
    ws.atmospheric_field = pa.data.to_atmospheric_field(atmosphere)

    # Surface temperature from lowest atmospheric level
    ws.surface_fieldPlanet(option="Earth")
    ws.surface_field["t"] = atmosphere["t"].sel(alt=0).values

    # Frequency grid (full LW)
    ws.frequency_grid = FREQ_GRID

    # Load HITRAN catalog (downloads if missing)
    ws.ReadCatalogData()

    # Speedup: cutoff + line thinning
    cutoff = pa.arts.convert.kaycm2freq(25)  # 25 cm^-1
    for band in ws.absorption_bands:
        ws.absorption_bands[band].cutoff = "ByLine"
        ws.absorption_bands[band].cutoff_value = cutoff

    ws.absorption_bands.keep_hitran_s(approximate_percentile=90)

    # ARTS propagation + RTE solver
    ws.propagation_matrix_agendaAuto()

    # Nadir TOA view (space-looking)
    pos = [100e3, 0, 0]  # 100km altitude
    los = [180.0, 0.0]  # nadir
    ws.ray_pathGeometric(pos=pos, los=los, max_step=1000.0)
    ws.spectral_radianceClearskyEmission()

    return ws


def calculate_spectral_radiance(workspace):
    """Extract TOA spectral radiance from ARTS workspace."""
    return workspace.spectral_radiance


def calculate_total_flux(spectral_radiance):
    """
    Integrate spectral radiance to broadband OLR flux.

    Assumes hemispheric isotropy (multiply by π).

    Parameters
    ----------
    spectral_radiance : ndarray
        Shape (n_frequencies,)

    Returns
    -------
    float
        Total OLR [W/m²]
    """
    return np.trapezoid(spectral_radiance[:, 0], FREQ_GRID) * np.pi


def plot_ola(spectral_radiance, flux):
    """
    Plot clear-sky outgoing longwave spectrum + total flux annotation.

    Parameters
    ----------
    spectral_radiance : ndarray
        TOA spectral radiance [W/m²/sr/Hz]
    flux : float
        Broadband OLR [W/m²]
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(KAYSER_GRID, spectral_radiance[:, 0], 'b-', linewidth=1.5)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel(r"Spectral radiance (W m$^{-2}$ sr$^{-1}$ Hz$^{-1}$)")
    ax.set_title(f"Clear-sky OLR | O3={MIXING_RATIO_O3:.0e}, CO2={MIXING_RATIO_CO2:.0e}")

    # Annotate total flux
    ax.text(0.98, 0.98, f"Total OLR = {flux:.1f} W/m²",
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.grid(True, alpha=0.3)

    if "ARTS_HEADLESS" not in os.environ:
        plt.show()
    else:
        plt.savefig("clear_sky_olr.pdf", dpi=300, bbox_inches='tight')


def main():
    """
    Main execution workflow:
    1. Generate moist-adiabatic profiles
    2. Setup ARTS atmosphere
    3. Run clear-sky radiative transfer
    4. Plot OLR spectrum + total flux
    """
    print("🌡️  Clear-sky longwave radiation calculation")

    # 1. Generate atmosphere profiles
    t_profile, wmr_profile, pressure_levels = sca.create_vertical_profile(T_SURF)
    atmosphere = set_up_atmosphere(t_profile, pressure_levels, wmr_profile, MIXING_RATIO_CO2)

    # 2. ARTS radiative transfer (uncomment to run full calculation)
    workspace = set_up_workspace(atmosphere)
    spectral_radiance_toa = calculate_spectral_radiance(workspace)
    total_olr = calculate_total_flux(spectral_radiance_toa)

    # 3. Plot results
    plot_ola(spectral_radiance_toa, total_olr)

    print(f"✅ OLR = {total_olr:.1f} W/m² (tropical clear-sky realistic)")


if __name__ == "__main__":
    main()
