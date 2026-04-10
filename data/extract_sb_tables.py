#!/usr/bin/env python3
"""Extract Seltzer-Berger cross section tables from G4EMLOW8.6.1 for the 10 NASA materials.

Output: data/sb_elements.json
Format: { "Z": { "x_kappa": [...32...], "y_logT": [...57...], "chi": [[...32...], ...57...] } }

The scaled DCS chi(T, kappa) relates to the physical cross section via:
    dσ/dk [cm²/MeV] = Z² × chi(T, κ) × (16α r₀²/3) / k
    where κ = k/T (reduced photon energy), T in MeV.

Reference: Seltzer & Berger, Atom. Data Nucl. Data Tables 35, 345-418 (1986).
Data from EMLOW 8.6.1 (GEANT4), path: brem_SB/br{Z}
"""

from __future__ import annotations

import json
from pathlib import Path

EMLOW_BREM_SB: Path = Path(
    "/opt/homebrew/Caskroom/miniforge/base/pkgs"
    "/geant4-data-emlow-8.6.1-hd8ed1ab_0/share/Geant4/data/EMLOW8.6.1/brem_SB"
)

# Z -> symbol for NASA materials
NASA_Z: dict[int, str] = {
    12: "Mg",
    13: "Al",
    22: "Ti",
    25: "Mn",
    26: "Fe",
    28: "Ni",
    29: "Cu",
    74: "W",
    79: "Au",
    82: "Pb",
}

OUTPUT: Path = Path(__file__).parent / "sb_elements.json"


def parse_sb_file(path: Path) -> dict[str, object]:
    """Parse a brem_SB br{Z} file into arrays."""
    lines = path.read_text().splitlines()
    # Line 0: "type nx ny" (e.g. "4 32 57")
    parts = lines[0].split()
    nx = int(parts[1])
    ny = int(parts[2])
    # Line 1: x-values (kappa = k/T), nx floats
    x_kappa = [float(v) for v in lines[1].split()]
    assert len(x_kappa) == nx, f"Expected {nx} x-values, got {len(x_kappa)}"
    # Line 2: y-values (ln(T/MeV)), ny floats
    y_log_t = [float(v) for v in lines[2].split()]
    assert len(y_log_t) == ny, f"Expected {ny} y-values, got {len(y_log_t)}"
    # Lines 3..3+ny-1: data rows, each with nx values
    chi: list[list[float]] = []
    for i in range(ny):
        row = [float(v) for v in lines[3 + i].split()]
        assert len(row) == nx, f"Row {i}: expected {nx} values, got {len(row)}"
        chi.append(row)
    return {"x_kappa": x_kappa, "y_logT": y_log_t, "chi": chi, "nx": nx, "ny": ny}


def main() -> None:
    result: dict[str, object] = {}
    for z, sym in NASA_Z.items():
        src = EMLOW_BREM_SB / f"br{z}"
        if not src.exists():
            print(f"  MISSING: {src}")
            continue
        data = parse_sb_file(src)
        result[str(z)] = {"symbol": sym, **data}
        print(
            f"  Z={z:2d} ({sym:2s}): nx={data['nx']}, ny={data['ny']}, "
            f"logT=[{data['y_logT'][0]:.3f}, {data['y_logT'][-1]:.3f}], "
            f"kappa=[{data['x_kappa'][0]:.6g}, {data['x_kappa'][-1]:.6g}]"
        )
    OUTPUT.write_text(json.dumps(result, separators=(",", ":")))
    print(f"\nWrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
