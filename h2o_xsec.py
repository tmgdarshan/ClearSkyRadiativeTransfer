"""
Calculate H2O absorption cross-sections using ARTS line-by-line data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyarts3 as pyarts


def calculate_h2o_cross_section():
    """
    Compute H2O absorption cross-section (10-2400 cm^-1) at 1000hPa, 295K, VMR=0.02.

    Returns
    -------
    f_grid_kayser : ndarray
        Frequency grid [cm^-1]
    xsec : ndarray
        Absorption cross-section [cm^2 molecule^-1]
    """
    # 1) Prepare ARTS workspace
    pyarts.data.download()
    ws = pyarts.workspace.Workspace()

    # 2) Set up absorption species and read catalog data
    ws.abs_speciesSet(species=["H2O, H2O-SelfContCKDMT400, H2O-ForeignContCKDMT400"])
    ws.ReadXML(ws.predefined_model_data, "model/mt_ckd_4.0/H2O.xml")
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename="lines/")
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)

    # 3) Set up line-by-line calculation
    ws.lbl_checkedCalc()
    ws.propmat_clearsky_agendaAuto()

    # 4) Initialize required workspace variables
    ws.stokes_dim = 1
    ws.jacobian_quantities = []
    ws.select_abs_species = []
    ws.rtp_mag = []
    ws.rtp_los = []
    ws.rtp_nlte = pyarts.arts.EnergyLevelMap()

    # 5) Set up frequency grid and atmospheric conditions
    f_grid_kayser = np.linspace(10, 2400, 30000)
    ws.f_grid = pyarts.arts.convert.kaycm2freq(f_grid_kayser)
    ws.rtp_pressure = 1000e2
    ws.rtp_temperature = 295
    ws.rtp_vmr = [0.02]

    # 6) Calculate absorption
    ws.AgendaExecute(a=ws.propmat_clearsky_agenda)

    # Convert to cross-section [cm^2]
    xsec = ws.propmat_clearsky.value.data.flatten() / (
            ws.rtp_vmr.value[0] * ws.rtp_pressure.value
            / (pyarts.arts.constants.k * ws.rtp_temperature.value)
    ) * 10000

    return f_grid_kayser, xsec


def plot_cross_section(f_grid_kayser, xsec):
    """
    Plot H2O cross-section (raw + smoothed).
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Raw data
    ax.semilogy(f_grid_kayser, xsec, lw=0.2, alpha=0.5, color="#932667")

    # Smoothed (rolling mean)
    def rolling_mean(x, w=1000):
        return np.convolve(x, np.ones(w) / w, "valid")

    ax.semilogy(
        rolling_mean(f_grid_kayser),
        10 ** rolling_mean(np.log10(xsec)),
        lw=2,
        color="#932667"
    )

    ax.set_xlabel("Wavenumber / cm$^{-1}$")
    ax.set_ylabel("Absorption cross section / cm$^2$ molecules$^{-1}$")
    ax.set_xlim(f_grid_kayser.min(), f_grid_kayser.max())
    ax.spines[["right", "top"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("h2o-xsec.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """
    Compute and plot H2O absorption cross-section.
    """
    f_grid_kayser, xsec = calculate_h2o_cross_section()
    plot_cross_section(f_grid_kayser, xsec)
    print(f"✅ H2O cross-section computed: {xsec.size} frequencies")


if __name__ == "__main__":
    main()
