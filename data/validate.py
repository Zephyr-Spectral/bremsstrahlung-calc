#!/usr/bin/env python3
"""NASA TN D-4755 validation script.

Compares calculated thick-target bremsstrahlung against digitized NASA
tables for all 10 materials x 5 energies x 3 angles.  Reports per-material
RMS deviation and flags any point > 3x deviation.

Usage:
    python data/validate.py              # summary table
    python data/validate.py --verbose    # per-point detail
    python data/validate.py --json       # JSON artifact for CI
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from server.physics.interpolation import interpolate_nasa_spectrum
from server.physics.thick_target import thick_target_intensity

log = logging.getLogger(__name__)

# NASA grid points
ENERGIES = config.NASA_ELECTRON_ENERGIES_MEV  # [0.50, 0.75, 1.00, 2.00, 3.00]
ANGLES = config.NASA_DETECTION_ANGLES_DEG  # [0.0, 30.0, 60.0]
MATERIALS = config.NASA_MATERIALS


def validate_single(
    symbol: str,
    electron_energy: float,
    angle_deg: float,
    n_slabs: int = 50,
) -> dict[str, float | list[float]]:
    """Compare calculated vs NASA at one grid point.

    Returns dict with k values, calculated, NASA, and per-point ratios.
    """
    mat = MATERIALS[symbol]
    z = int(mat["Z"])
    a_val = float(mat["A"])
    density = float(mat["density"])

    k_nasa, i_nasa = interpolate_nasa_spectrum(electron_energy, angle_deg, symbol)
    if not k_nasa:
        return {"error": "no NASA data", "n_points": 0}

    ratios: list[float] = []
    calc_vals: list[float] = []

    for k_val, nasa_val in zip(k_nasa, i_nasa, strict=False):
        if nasa_val <= 0 or k_val <= 0:
            continue
        calc = thick_target_intensity(
            electron_energy, k_val, angle_deg, z, a_val, density, symbol, n_slabs=n_slabs
        )
        calc_vals.append(calc)
        ratio = calc / nasa_val if nasa_val > 0 else 0.0
        ratios.append(ratio)

    if not ratios:
        return {"error": "no valid points", "n_points": 0}

    import math

    mean_ratio = sum(ratios) / len(ratios)
    rms_dev = math.sqrt(sum((r - 1.0) ** 2 for r in ratios) / len(ratios))
    max_ratio = max(ratios)
    min_ratio = min(ratios)

    return {
        "n_points": len(ratios),
        "mean_ratio": round(mean_ratio, 3),
        "rms_deviation": round(rms_dev, 3),
        "min_ratio": round(min_ratio, 3),
        "max_ratio": round(max_ratio, 3),
    }


def run_full_sweep(
    n_slabs: int = 50,
    verbose: bool = False,
) -> dict[str, object]:
    """Run validation across all NASA grid points.

    Returns structured results dict.
    """
    results: dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_slabs": n_slabs,
        "materials": {},
    }

    total_points = 0
    total_flags = 0

    for symbol in MATERIALS:
        mat_results: dict[str, object] = {"grid_points": []}
        mat_ratios: list[float] = []

        for energy in ENERGIES:
            for angle in ANGLES:
                result = validate_single(symbol, energy, angle, n_slabs)
                entry = {
                    "energy_mev": energy,
                    "angle_deg": angle,
                    **result,
                }

                if isinstance(result.get("mean_ratio"), float):
                    mat_ratios.append(result["mean_ratio"])
                    total_points += int(result.get("n_points", 0))

                    flagged = result.get("max_ratio", 0) > 3.0 or result.get("min_ratio", 1) < 0.33
                    if flagged:
                        total_flags += 1
                        entry["FLAGGED"] = True

                    if verbose:
                        flag = " *** FLAGGED" if flagged else ""
                        ratio_str = f"{result['mean_ratio']:.2f}"
                        sys.stdout.write(
                            f"  {symbol:3s} E={energy:.2f} phi={angle:4.0f}"
                            f"  ratio={ratio_str:>5s}"
                            f"  rms={result['rms_deviation']:.3f}{flag}\n"
                        )

                mat_results["grid_points"].append(entry)  # type: ignore[union-attr]

        if mat_ratios:
            import math

            overall_mean = sum(mat_ratios) / len(mat_ratios)
            overall_rms = math.sqrt(sum((r - 1.0) ** 2 for r in mat_ratios) / len(mat_ratios))
            mat_results["overall_mean_ratio"] = round(overall_mean, 3)
            mat_results["overall_rms_deviation"] = round(overall_rms, 3)

        results["materials"][symbol] = mat_results  # type: ignore[index]

    results["total_points"] = total_points
    results["total_flags"] = total_flags

    return results


def print_summary(results: dict[str, object]) -> None:
    """Print human-readable summary table."""
    sys.stdout.write("\n" + "=" * 65 + "\n")
    sys.stdout.write("  NASA TN D-4755 Validation Summary\n")
    sys.stdout.write("=" * 65 + "\n\n")

    sys.stdout.write(f"{'Material':>10s}  {'Mean Ratio':>12s}  {'RMS Dev':>10s}  {'Status':>8s}\n")
    sys.stdout.write("-" * 50 + "\n")

    materials = results.get("materials", {})
    if not isinstance(materials, dict):
        return

    for symbol, mat_data in materials.items():
        if not isinstance(mat_data, dict):
            continue
        mean = mat_data.get("overall_mean_ratio", 0)
        rms = mat_data.get("overall_rms_deviation", 0)
        status = "PASS" if isinstance(rms, float) and rms < 1.0 else "REVIEW"
        sys.stdout.write(f"{symbol:>10s}  {mean:>12.3f}  {rms:>10.3f}  {status:>8s}\n")

    sys.stdout.write(
        f"\nTotal points: {results.get('total_points', 0)}"
        f"  |  Flagged (>3x): {results.get('total_flags', 0)}\n\n"
    )


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Validate against NASA TN D-4755")
    parser.add_argument("--verbose", action="store_true", help="Show per-point detail")
    parser.add_argument("--json", action="store_true", help="Output JSON artifact")
    parser.add_argument("--slabs", type=int, default=30, help="Integration slabs (default 30)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    sys.stdout.write("Running NASA TN D-4755 validation sweep...\n")
    sys.stdout.write(f"  Materials: {list(MATERIALS.keys())}\n")
    sys.stdout.write(f"  Energies:  {ENERGIES} MeV\n")
    sys.stdout.write(f"  Angles:    {ANGLES} deg\n")
    sys.stdout.write(f"  Slabs:     {args.slabs}\n\n")

    t0 = time.time()
    results = run_full_sweep(n_slabs=args.slabs, verbose=args.verbose)
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    if args.json:
        output_path = Path("data/validation_results.json")
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        sys.stdout.write(f"\nJSON written to {output_path}\n")
    else:
        print_summary(results)

    sys.stdout.write(f"Elapsed: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
