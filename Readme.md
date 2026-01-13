# Clear-Sky Radiative Transfer Model

**Using ARTS3 to calculate clear-sky upwelling longwave radiation at top of atmosphere (TOA) with a tropical atmosphere containing H₂O, CO₂, O₃ which includes the water vapor continuum.**

Single-column model computing outgoing longwave radiation (OLR) using ARTS/pyarts3.

## Installation (Mamba Required)

```bash
git clone https://github.com/YOURUSERNAME/clear-sky-radiative-transfer.git
cd clear-sky-radiative-transfer

# Install pyarts3 + dependencies
mamba install -c rttools-dev pyarts3 numpy matplotlib xarray typhon

# Or install requirements one-by-one
pip install numpy matplotlib xarray typhon
mamba install -c rttools-dev pyarts3
