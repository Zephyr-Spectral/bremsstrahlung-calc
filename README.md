# Bremsstrahlung Calculator

Interactive thick-target bremsstrahlung calculator based on NASA TN D-4755 (Powell, 1968).

## Overview

Computes angular and energy distributions of bremsstrahlung radiation produced by MeV electrons incident on thick metal targets. Combines first-principles physics (Koch & Motz cross sections, Berger multiple scattering, NIST XCOM photon attenuation) with digitized NASA tabulated data for 10 metals.

## Physics

### Cross section
Koch & Motz 2BN doubly-differential cross section (Rev. Mod. Phys. 31, 920, 1959) with 2BS Schiff screening fallback. The 2BN is the full unscreened Born approximation valid at all emission angles.

### Thick-target integration (Powell eq. 14)
Sums thin-slab contributions over the electron slowing-down path:

- Bethe collision + radiative stopping power (no density effect, matching Powell eq. 8)
- Berger Legendre polynomial expansion for multiple electron scattering
- Spherical triangle geometry for scattered-electron to photon-angle conversion
- NIST XCOM photon mass attenuation with K/L/M absorption edge structure
- Taylor buildup factor for photon transmission
- Backscatter correction (Wright & Trump empirical fit)
- Electron-electron bremsstrahlung: Z^2 -> Z(Z+1)

### Materials
- NASA 10: Mg, Al, Ti, Mn, Fe, Ni, Cu, W, Au, Pb (with tabulated reference data)
- Extended: SS304, SS316 (effective-Z composites, calculation only)

### Accuracy vs NASA tables (1 MeV, k=0.25 MeV)
| Angle | Low-Z (Al, Fe, Cu) | High-Z (W, Pb) |
|-------|---------------------|-----------------|
| 0 deg | 1.5-1.7x | 1.9x |
| 30 deg | 0.94-1.04x | 1.3-1.4x |
| 60 deg | 0.63x | 0.82-0.90x |

## Quick start

```bash
# Use existing science_venv (Python 3.12+, FastAPI, scipy, numpy)
source ~/science_venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run the server
uvicorn server.main:app --reload --port 8001

# Open in browser
open http://127.0.0.1:8001
```

## Quality gates

```bash
ruff check . && ruff format --check .    # linting + formatting
mypy --strict server/ config.py          # type checking
pytest tests/ -v -m 'not slow'           # unit tests (55 tests)
python data/validate.py                  # NASA data validation sweep
```

## Project structure

```
bremsstrahlung-calc/
  config.py                 # All constants, material data, paths
  server/
    main.py                 # FastAPI app
    data_access.py          # NASA JSON data loader
    physics/
      bethe_heitler.py      # Koch & Motz 2BN/2BS cross sections
      thick_target.py       # Powell eq. 14 integration engine
      scattering.py         # Berger multiple scattering (Legendre)
      stopping_power.py     # Bethe collision + radiative
      electron_range.py     # CSDA range integration
      attenuation.py        # NIST XCOM photon attenuation
      interpolation.py      # NASA table interpolation
    api/
      spectrum.py           # /api/spectrum/* endpoints
      materials.py          # /api/materials/* endpoints
      validation.py         # /api/validation/* endpoints
  data/
    nasa_tnd4755/tables.json  # Digitized NASA Tables I-XIV
    xcom_elements.json        # NIST XCOM photon cross sections (Z=1-92)
    validate.py               # Validation sweep script
  static/                     # Dark/light theme CSS, Plotly.js frontend
  templates/index.html        # SPA with 7 tabbed views
  tests/                      # pytest suite (55 tests)
```

## References

1. Powell CA Jr., "Tables of energy and angular distributions of thick target bremsstrahlung in metals," NASA TN D-4755, 1968.
2. Koch HW & Motz JW, "Bremsstrahlung cross-section formulas and related data," Rev. Mod. Phys. 31, 920-955, 1959.
3. Berger MJ & Seltzer SM, "Tables of energy losses and ranges of electrons and positrons," NASA SP-3012, 1964.
4. NIST XCOM photon cross sections database.
