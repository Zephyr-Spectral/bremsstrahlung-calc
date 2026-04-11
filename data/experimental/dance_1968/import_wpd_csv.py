"""Import WebPlotDigitizer CSV exports into dance_1968.json.

WebPlotDigitizer exports CSVs with two columns: X, Y
For our plots:
  X = photon energy (MeV)
  Y = intensity k*dn/dk/dOmega (MeV/MeV/sr/electron)

For Fig 2 (Fe), the axes are linear in energy, log in intensity.
For Fig 1 (Al 2 MeV), the axes are linear in energy, log in intensity.
For Fig 3, x-axis is k/T0 (normalized), y-axis is log intensity.

Naming convention for CSV files in wpd_exports/:
  {material}_{energy_mev}MeV_{angle}deg.csv

Examples:
  Al_2.0MeV_0deg.csv
  Fe_0.5MeV_30deg.csv
  Fe_2.8MeV_150deg.csv

For Fig 3 (normalized x-axis), use suffix _norm:
  Al_0.5MeV_fwd_norm.csv   (forward hemisphere 0-90 deg integrated)
  Fe_1.0MeV_full_norm.csv  (full space 0-180 deg integrated)

Usage:
    python import_wpd_csv.py
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent
WPD_DIR = DATA_DIR / "wpd_exports"
JSON_PATH = DATA_DIR / "dance_1968.json"


def import_all() -> None:
    """Import all CSV files from wpd_exports/ into dance_1968.json."""
    if not WPD_DIR.exists():
        WPD_DIR.mkdir()
        print(f"Created {WPD_DIR}/")  # noqa: T201
        print("Place WebPlotDigitizer CSV exports here with naming convention:")  # noqa: T201
        print("  {Material}_{Energy}MeV_{Angle}deg.csv")  # noqa: T201
        print("  e.g., Fe_1.0MeV_30deg.csv")  # noqa: T201
        return

    csv_files = sorted(WPD_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {WPD_DIR}/")  # noqa: T201
        return

    with JSON_PATH.open() as f:
        data = json.load(f)

    imported = 0
    for csv_path in csv_files:
        name = csv_path.stem  # e.g., "Fe_1.0MeV_30deg"
        parts = name.split("_")
        if len(parts) < 3:
            print(f"  Skipping {csv_path.name}: invalid name format")  # noqa: T201
            continue

        material = parts[0]
        energy_str = parts[1].replace("MeV", "")
        angle_str = parts[2].replace("deg", "")

        # Read CSV
        k_values: list[float] = []
        i_values: list[float] = []
        with csv_path.open() as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    k = float(row[0])
                    intensity = float(row[1])
                    if k > 0 and intensity > 0:
                        k_values.append(round(k, 6))
                        i_values.append(round(intensity, 6))
                except ValueError:
                    continue

        if not k_values:
            print(f"  Skipping {csv_path.name}: no valid data")  # noqa: T201
            continue

        # Format energy key to match JSON structure
        energy_key = f"{float(energy_str):.2f}"

        # Insert into JSON
        if material not in data["spectra"]:
            print(f"  Skipping {csv_path.name}: unknown material {material}")  # noqa: T201
            continue
        if energy_key not in data["spectra"][material]:
            print(f"  Skipping {csv_path.name}: unknown energy {energy_key}")  # noqa: T201
            continue
        if angle_str not in data["spectra"][material][energy_key]["angles"]:
            print(f"  Skipping {csv_path.name}: unknown angle {angle_str}")  # noqa: T201
            continue

        data["spectra"][material][energy_key]["angles"][angle_str] = {
            "photon_energy_mev": k_values,
            "intensity": i_values,
        }
        imported += 1
        print(f"  Imported {csv_path.name}: {len(k_values)} points")  # noqa: T201

    # Write updated JSON
    with JSON_PATH.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"\nImported {imported} datasets into {JSON_PATH.name}")  # noqa: T201


if __name__ == "__main__":
    import_all()
