# Bremsstrahlung Calculator

## Architecture

FastAPI web app for thick-target bremsstrahlung calculations based on NASA TN D-4755 (Powell, 1968).

### Key directories
- `config.py` — all constants, material data, physics constants, paths (single source of truth)
- `server/physics/` — six physics modules implementing Powell's thick-target method
- `server/api/` — REST API routers (spectrum, materials, validation)
- `server/data_access.py` — loads and caches NASA JSON data
- `data/nasa_tnd4755/tables.json` — digitized NASA tables I-XIV
- `static/` — dark-theme CSS, vanilla JS with Plotly.js
- `templates/index.html` — SPA shell with 7 tabbed views

### Running
```bash
pip install -e ".[dev]"
uvicorn server.main:app --reload --port 8001
```

### Testing
```bash
ruff check . && ruff format --check .
mypy --strict server/ config.py
pytest tests/ -v -m 'not slow'
python data/validate.py  # NASA data validation
```

## Standards
- Python 3.12+, `from __future__ import annotations` on every module
- Type hints on all function signatures, `mypy --strict` must pass
- No hardcoded values — all tunables in `config.py` or function params with defaults
- `logging` module only (no print), enforced by T20 rule
- Ruff full rule set must pass
- pytest 80%+ coverage minimum
- All physics functions require edge case tests

## Physics
- Bethe-Heitler thin-target cross section (Koch & Motz eq. 2BN)
- Multiple electron scattering (Berger Legendre series)
- Electron backscatter correction
- Electron-electron bremsstrahlung: Z² → Z(Z+1)
- Photon absorption + Taylor buildup
- Thick-target integration over thin slabs (Powell eqs 1-14)

## Materials
NASA 10: Mg, Al, Ti, Mn, Fe, Ni, Cu, W, Au, Pb + SS304, SS316 (effective-Z)
