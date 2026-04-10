# Bremsstrahlung Calculator

## Architecture

FastAPI web app for thick-target bremsstrahlung calculations based on NASA TN D-4755 (Powell, 1968).

### Key directories
- `config.py` — all constants, material data, physics constants, paths (single source of truth)
- `server/physics/` — seven physics modules implementing Powell's thick-target method
- `server/api/` — REST API routers (spectrum, materials, validation)
- `server/data_access.py` — loads and caches NASA JSON data
- `data/nasa_tnd4755/tables.json` — digitized NASA tables I-XIV
- `data/xcom_elements.json` — NIST XCOM photon cross sections (Z=1-92, full edge structure)
- `data/validate.py` — NASA comparison validation sweep
- `static/` — dark/light theme CSS, vanilla JS with Plotly.js
- `templates/index.html` — SPA shell with 7 tabbed views

### Running
```bash
source ~/science_venv/bin/activate
uvicorn server.main:app --reload --port 8001
```

### Testing
```bash
ruff check . && ruff format --check .
mypy --strict server/ config.py
pytest tests/ -v -m 'not slow'
python data/validate.py            # NASA data validation
python data/validate.py --json     # generate JSON artifact
```

## Standards
- Python 3.12+, `from __future__ import annotations` on every module
- Type hints on all function signatures, `mypy --strict` must pass
- No hardcoded values — all tunables in `config.py` or function params with defaults
- `logging` module only (no print), enforced by T20 rule
- Ruff full rule set must pass (E, F, W, I, UP, B, SIM, A, T20, S, C90, PT, RUF, PERF, LOG)
- pytest 80%+ coverage minimum
- All physics functions require edge case tests
- `data/validate.py` must run clean before release

## Physics implementation

### Cross section (bethe_heitler.py)
- **Primary**: Koch & Motz 2BN (full unscreened Born, valid at all angles)
- **Fallback**: Koch & Motz 2BS (Schiff with Thomas-Fermi screening)
- 2BN falls back to 2BS only when numerical cancellation produces negative bracket

### Thick-target engine (thick_target.py)
Powell eq. 14: triple sum over energy slabs (i), scattering angles (alpha), azimuthal angles (gamma):
- dE/dt from Bethe formula WITHOUT density effect (matches Powell eq. 8)
- Berger Legendre scattering (eq. 4) with corrected Omega_0 = 4 pi r0^2 N_A Z(Z+1) / (A p^2 beta^2)
- Spherical triangle (eq. 3) for theta_0 from detection and scattering angles
- NIST XCOM photon attenuation with K/L/M edge structure + Taylor buildup
- Backscatter correction (1-W) from Wright & Trump fit
- Z(Z+1) electron-electron bremsstrahlung correction

### Key physics constants (config.py)
All in CGS/MeV units matching the NASA paper conventions. Critical values:
- ELECTRON_MASS_MEV = 0.510998950
- CLASSICAL_ELECTRON_RADIUS_CM = 2.8179403e-13
- ALPHA_FINE = 1/137.035999

### Accuracy vs NASA (1 MeV, k=0.25 MeV)
- 0 deg: 1.5-1.9x (forward scattering peak slightly over-predicted)
- 30 deg: 0.94-1.04x (excellent agreement)
- 60 deg: 0.63-0.90x (angular quadrature resolution limited)

## Materials
- NASA 10: Mg, Al, Ti, Mn, Fe, Ni, Cu, W, Au, Pb (with tabulated reference data)
- Extended: SS304, SS316 (effective-Z composites, calculation only)
- XCOM data available for all Z=1-92

## API endpoints
- `GET /api/spectrum/calculate` — spectrum at fixed angle
- `GET /api/spectrum/angular` — angular distribution at fixed photon energy
- `GET /api/spectrum/integrated` — angle-integrated spectrum
- `GET /api/spectrum/compare` — multi-material comparison
- `GET /api/spectrum/heatmap` — 2D intensity grid (photon energy x angle)
- `GET /api/spectrum/nasa-data` — raw NASA grid-point data
- `GET /api/materials` — material list with properties
- `GET /api/materials/{symbol}/stopping-power` — stopping power curve
- `GET /api/materials/{symbol}/range` — CSDA range curve
- `GET /api/validation/nasa-comparison` — calculated vs NASA at grid point
