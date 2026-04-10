#!/usr/bin/env python3
"""Generate pre-computed DDCS tables from BremsLib v2.0 for all elements.

Runs Interpolate_DCS (Fortran) for each element Z=1..100 at a grid of
(T1, k/T1, theta) values covering the thick-target bremsstrahlung range
(T1 = 0.05-3.0 MeV). Output is saved as a single .npz file for fast
loading by the Python thick-target code.

Units in the output:
    DDCS [cm^2 / (sr MeV atom)]  — ready for direct use in the thick-target integral.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INTERPOLATE_DCS = Path(__file__).parent / "build" / "Interpolate_DCS"
BREMSLIB_DIR = Path(__file__).parent / "BremsLib_v2.0"
OUTPUT_FILE = Path(__file__).parent / "ddcs_all_elements.npz"

# Electron kinetic energies [MeV] — covers 50 keV to 30 MeV (BremsLib limit)
T1_GRID = [
    0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00,
    4.00, 5.00, 6.00, 8.00, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0,
]

# Reduced photon energies k/T1 (avoid 0 and 1 endpoints)
KAPPA_GRID = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
]

# Photon emission angles [degrees]
THETA_GRID = [
    0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0,
    25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
    100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0,
]


def run_interpolate_dcs(
    z: int, t1_mev: float, kappa_values: list[float], theta_file: Path
) -> dict[float, list[float]] | None:
    """Run Interpolate_DCS for one (Z, T1) and multiple k/T1, returning DDCS[cm2/sr/MeV].

    Returns dict mapping kappa -> list of DDCS values at THETA_GRID angles,
    or None if the run fails.
    """
    result: dict[float, list[float]] = {}

    for kappa in kappa_values:
        k_mev = kappa * t1_mev
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Copy knot files to tmp
            for knot_file in Path(__file__).parent.glob("build/*.txt"):
                (tmpdir_path / knot_file.name).write_text(knot_file.read_text())
            # Copy default parameters
            for inp_file in Path(__file__).parent.glob("build/*.inp"):
                (tmpdir_path / inp_file.name).write_text(inp_file.read_text())

            cmd = [
                str(INTERPOLATE_DCS),
                f"Z={z}",
                f"T1={t1_mev}",
                f"k={k_mev}",
                f"theta_fileName={theta_file}",
                f"prefix_lib={BREMSLIB_DIR}/",
            ]

            proc = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(tmpdir_path), timeout=30
            )

            if proc.returncode != 0:
                log.warning("Z=%d T1=%.3f k/T1=%.2f failed: %s", z, t1_mev, kappa,
                            proc.stderr[:200] if proc.stderr else "unknown")
                result[kappa] = [0.0] * len(THETA_GRID)
                continue

            # Find the output file
            out_files = list(tmpdir_path.glob("Z=*_T1=*_k=*.txt"))
            if not out_files:
                log.warning("Z=%d T1=%.3f k/T1=%.2f: no output file", z, t1_mev, kappa)
                result[kappa] = [0.0] * len(THETA_GRID)
                continue

            # Parse output: columns are theta, DDCS_scaled, DDCS[cm2/sr/MeV], relErr
            ddcs_vals: list[float] = []
            for line in out_files[0].read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("theta"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    ddcs_vals.append(float(parts[2]))  # DDCS in cm2/sr/MeV
                else:
                    ddcs_vals.append(0.0)

            if len(ddcs_vals) != len(THETA_GRID):
                log.warning(
                    "Z=%d T1=%.3f k/T1=%.2f: got %d angles, expected %d",
                    z, t1_mev, kappa, len(ddcs_vals), len(THETA_GRID),
                )
                # Pad or truncate
                while len(ddcs_vals) < len(THETA_GRID):
                    ddcs_vals.append(0.0)
                ddcs_vals = ddcs_vals[: len(THETA_GRID)]

            result[kappa] = ddcs_vals

    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not INTERPOLATE_DCS.exists():
        log.error("Interpolate_DCS not found at %s — compile first", INTERPOLATE_DCS)
        sys.exit(1)
    if not BREMSLIB_DIR.exists():
        log.error("BremsLib data not found at %s — extract first", BREMSLIB_DIR)
        sys.exit(1)

    # Write theta file
    theta_file = Path(__file__).parent / "build" / "theta_grid.txt"
    with theta_file.open("w") as f:
        f.write("theta(deg)\n")
        for th in THETA_GRID:
            f.write(f"{th}\n")

    n_t = len(T1_GRID)
    n_k = len(KAPPA_GRID)
    n_th = len(THETA_GRID)

    log.info("Grid: %d T1 x %d kappa x %d theta = %d points per element",
             n_t, n_k, n_th, n_t * n_k * n_th)
    log.info("Generating tables for Z=1..100...")

    # 4D array: (Z, T1, kappa, theta)
    all_ddcs = np.zeros((100, n_t, n_k, n_th), dtype=np.float64)

    for z in range(1, 101):
        log.info("  Z=%3d ...", z)
        for i_t, t1 in enumerate(T1_GRID):
            data = run_interpolate_dcs(z, t1, KAPPA_GRID, theta_file)
            if data is None:
                continue
            for i_k, kappa in enumerate(KAPPA_GRID):
                if kappa in data:
                    all_ddcs[z - 1, i_t, i_k, :] = data[kappa]

    # Save as compressed npz
    np.savez_compressed(
        OUTPUT_FILE,
        ddcs=all_ddcs,          # (100, n_t, n_k, n_th) cm2/sr/MeV
        t1_grid=np.array(T1_GRID),
        kappa_grid=np.array(KAPPA_GRID),
        theta_grid=np.array(THETA_GRID),
    )

    size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    log.info("Wrote %s (%.1f MB)", OUTPUT_FILE, size_mb)
    log.info("Shape: %s, dtype: %s", all_ddcs.shape, all_ddcs.dtype)

    # Quick sanity check: Cu at 1 MeV, k/T=0.25, theta=0
    i_t = T1_GRID.index(1.0)
    i_k = KAPPA_GRID.index(0.25)
    val = all_ddcs[28, i_t, i_k, 0]  # Z=29 -> index 28
    log.info("Sanity: Cu(Z=29), T1=1.0 MeV, k/T=0.25, theta=0: DDCS=%.3e cm2/sr/MeV", val)


if __name__ == "__main__":
    main()
