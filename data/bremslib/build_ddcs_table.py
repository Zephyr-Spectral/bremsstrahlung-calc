#!/usr/bin/env python3
"""Build pre-computed DDCS table from raw BremsLib v2.0 text files.

Reads all 86K DDCS_*.txt files in one pass, resamples to a common angular
grid, and saves a single compressed .npz for fast runtime loading.

No Fortran needed — direct Python read of the BremsLib grid data.

Output arrays in the .npz:
    ddcs_scaled : float32, shape (100, n_T1, n_kappa, n_theta)
        k/(Z^2) * d^2sigma/(dk dOmega) in mb/sr
    t1_grid     : float64, sorted electron energies [MeV]
    kappa_grid  : float64, sorted k/T1 values
    theta_grid  : float64, common resampled angles [degrees]

To recover physical DDCS:
    d^2sigma/(dk dOmega) [cm^2/sr/MeV] = ddcs_scaled * Z^2 / k * 1e-27
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DDCS_DIR = Path(__file__).parent / "BremsLib_v2.0" / "DDCS"
OUTPUT = Path(__file__).parent / "ddcs_all_elements.npz"

# Common theta grid for resampling (27 points: fine near forward, coarser at back)
THETA_COMMON = np.array([
    0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0,
    25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
    100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0,
], dtype=np.float64)

# Regex: DDCS_{Z}_{T1}_{k}.txt
FILE_RE = re.compile(
    r"^DDCS_(\d+)_([0-9.E+\-]+)_([0-9.E+\-]+)\.txt$"
)


def parse_ddcs_file(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a single DDCS file. Returns (theta_deg, ddcs_scaled) arrays."""
    thetas = []
    vals = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("theta"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            thetas.append(float(parts[0]))
            vals.append(float(parts[1]))
    if not thetas:
        return None
    return np.array(thetas), np.array(vals)


def resample_to_common(
    theta_raw: np.ndarray, ddcs_raw: np.ndarray
) -> np.ndarray:
    """Linear interpolation of DDCS from raw theta grid to THETA_COMMON."""
    return np.interp(THETA_COMMON, theta_raw, ddcs_raw)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not DDCS_DIR.exists():
        log.error("DDCS directory not found: %s", DDCS_DIR)
        sys.exit(1)

    t0 = time.perf_counter()

    # Scan all DDCS_*.txt files (NOT Born_appr subdirectory)
    all_files = sorted(DDCS_DIR.glob("DDCS_*.txt"))
    log.info("Found %d DDCS files", len(all_files))

    # First pass: discover all unique (Z, T1, kappa) values
    file_index: dict[tuple[int, float, float], Path] = {}
    all_z: set[int] = set()
    all_t1: set[float] = set()
    all_kappa: set[float] = set()

    for f in all_files:
        m = FILE_RE.match(f.name)
        if not m:
            continue
        z = int(m.group(1))
        t1 = float(m.group(2))
        k = float(m.group(3))
        kappa = k / t1 if t1 > 0 else 0.0
        # Round kappa to avoid floating point mess
        kappa = round(kappa, 6)
        file_index[(z, t1, kappa)] = f
        all_z.add(z)
        all_t1.add(t1)
        all_kappa.add(kappa)

    t1_grid = np.array(sorted(all_t1))
    kappa_grid = np.array(sorted(all_kappa))
    n_z = 100  # Z=1..100
    n_t1 = len(t1_grid)
    n_kappa = len(kappa_grid)
    n_theta = len(THETA_COMMON)

    log.info("Grid: Z=1..100, %d T1 values, %d kappa values, %d theta values",
             n_t1, n_kappa, n_theta)
    log.info("T1 range: %.4g - %.4g MeV", t1_grid[0], t1_grid[-1])
    log.info("kappa range: %.4g - %.4g", kappa_grid[0], kappa_grid[-1])
    log.info("Array shape: (%d, %d, %d, %d) = %.1f M values",
             n_z, n_t1, n_kappa, n_theta,
             n_z * n_t1 * n_kappa * n_theta / 1e6)

    # Build lookup for fast index access
    t1_to_idx = {t1: i for i, t1 in enumerate(t1_grid)}
    kappa_to_idx = {round(k, 6): i for i, k in enumerate(kappa_grid)}

    # Allocate output array (float32 to save space)
    ddcs = np.zeros((n_z, n_t1, n_kappa, n_theta), dtype=np.float32)

    # Second pass: read and resample each file
    n_ok = 0
    n_fail = 0
    for (z, t1, kappa), fpath in file_index.items():
        if z < 1 or z > 100:
            continue
        i_z = z - 1
        i_t = t1_to_idx.get(t1)
        i_k = kappa_to_idx.get(round(kappa, 6))
        if i_t is None or i_k is None:
            n_fail += 1
            continue

        result = parse_ddcs_file(fpath)
        if result is None:
            n_fail += 1
            continue

        theta_raw, ddcs_raw = result
        ddcs[i_z, i_t, i_k, :] = resample_to_common(theta_raw, ddcs_raw)
        n_ok += 1

    t1_elapsed = time.perf_counter() - t0
    log.info("Read %d files OK, %d failed, in %.1fs", n_ok, n_fail, t1_elapsed)

    # Save
    np.savez_compressed(
        OUTPUT,
        ddcs_scaled=ddcs,
        t1_grid=t1_grid,
        kappa_grid=kappa_grid,
        theta_grid=THETA_COMMON,
    )

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    log.info("Wrote %s (%.1f MB)", OUTPUT.name, size_mb)

    # Sanity check: Cu (Z=29), T1=1 MeV, kappa=0.2, theta=0
    i_z = 28
    i_t = np.searchsorted(t1_grid, 1.0)
    i_k = np.searchsorted(kappa_grid, 0.2)
    val = ddcs[i_z, i_t, i_k, 0]
    log.info("Sanity: Cu T1=1MeV kappa=0.2 theta=0: DDCS_scaled=%.4f mb/sr "
             "(expect ~22.8)", val)


if __name__ == "__main__":
    main()
