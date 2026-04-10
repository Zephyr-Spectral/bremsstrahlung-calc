#!/usr/bin/env python3
"""Post-process and plot Geant4 kill sphere results.

Reads the binary photon/electron files from the 10M event run and produces
publication-quality figures comparing Geant4 Monte Carlo with our semi-analytical
calculation and NASA TN D-4755 data.

Usage:
    python plot_results.py [prefix]
    Default prefix: fe_10M

Outputs:
    {prefix}_spectrum_0deg.png    — energy spectrum at 0 deg (forward)
    {prefix}_spectrum_30deg.png   — energy spectrum at 30 deg
    {prefix}_spectrum_60deg.png   — energy spectrum at 60 deg
    {prefix}_spectrum_90deg.png   — energy spectrum at 90 deg (perpendicular)
    {prefix}_angular.png          — angular distribution at several photon energies
    {prefix}_electron_angular.png — scattered electron angular distribution
    {prefix}_energy_balance.png   — energy balance pie chart
    {prefix}_all_angles.png       — multi-panel spectrum at 0, 30, 60, 90, 120, 150 deg
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "fe_10M"
PHOTON_FILE = Path(__file__).parent / f"{PREFIX}_photons.bin"
ELECTRON_FILE = Path(__file__).parent / f"{PREFIX}_electrons.bin"
SUMMARY_FILE = Path(__file__).parent / f"{PREFIX}_summary.txt"
OUT_DIR = Path(__file__).parent

# Plot style
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#333",
    "grid.alpha": 0.5,
    "legend.facecolor": "#16213e",
    "legend.edgecolor": "#444",
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 150,
})

C_G4 = "#50fa7b"       # green — Geant4
C_CALC = "#e94560"      # red — our calculation
C_NASA = "#00d2ff"      # cyan — NASA
C_ELEC = "#ff79c6"      # pink — electrons

# Material params for Fe
MAT = "Fe"
Z, A, RHO = 26, 55.845, 7.874
T0 = 3.0  # MeV


# ---------------------------------------------------------------------------
# Load binary data
# ---------------------------------------------------------------------------
def load_photons(path: Path) -> np.ndarray:
    """Load photon binary file. Returns structured array with k, theta, phi."""
    with path.open("rb") as f:
        header = struct.unpack("3i", f.read(12))
        n_events = header[0]
        dt = np.dtype([("k", "f4"), ("theta", "f4"), ("phi", "f4")])
        data = np.fromfile(f, dtype=dt)
    print(f"Loaded {len(data)} photons from {path.name} ({n_events} events)")
    return data


def load_electrons(path: Path) -> np.ndarray:
    """Load electron binary file. Returns structured array."""
    with path.open("rb") as f:
        header = struct.unpack("3i", f.read(12))
        n_events = header[0]
        dt = np.dtype([
            ("k", "f4"), ("theta", "f4"), ("phi", "f4"),
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ])
        data = np.fromfile(f, dtype=dt)
    print(f"Loaded {len(data)} electrons from {path.name} ({n_events} events)")
    return data


def get_n_events(path: Path) -> int:
    """Read n_events from binary header."""
    with path.open("rb") as f:
        return struct.unpack("3i", f.read(12))[0]


# ---------------------------------------------------------------------------
# Spectrum binning
# ---------------------------------------------------------------------------
def bin_spectrum(
    photons: np.ndarray,
    angle_center: float,
    angle_half_width: float,
    n_events: int,
    n_bins: int = 35,
    k_min: float = 0.02,
    k_max: float = 2.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin photons into energy spectrum at a given angle.

    Returns (k_centers, intensity, rel_error).
    Intensity units: MeV / (MeV sr electron).
    """
    ang_lo = max(angle_center - angle_half_width, 0.0)
    ang_hi = min(angle_center + angle_half_width, 180.0)
    mask = (photons["theta"] >= ang_lo) & (photons["theta"] < ang_hi)
    k_sel = photons["k"][mask]

    cos_lo = np.cos(np.radians(ang_lo))
    cos_hi = np.cos(np.radians(ang_hi))
    solid_angle = 2.0 * np.pi * abs(cos_lo - cos_hi)

    bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    counts, _ = np.histogram(k_sel, bins=bin_edges)
    k_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    dk = np.diff(bin_edges)

    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = k_centers * counts / (n_events * dk * solid_angle)
        rel_err = np.where(counts > 0, 1.0 / np.sqrt(counts), 0.0)

    return k_centers, intensity, rel_err


def angular_distribution(
    photons: np.ndarray,
    k_center: float,
    k_half_width: float,
    n_events: int,
    n_angle_bins: int = 36,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin photons into angular distribution at a given photon energy.

    Returns (theta_centers, intensity, rel_error).
    """
    mask = (photons["k"] >= k_center - k_half_width) & (photons["k"] < k_center + k_half_width)
    theta_sel = photons["theta"][mask]
    dk = 2 * k_half_width

    bin_edges = np.linspace(0, 180, n_angle_bins + 1)
    counts, _ = np.histogram(theta_sel, bins=bin_edges)
    theta_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Solid angle per bin
    cos_lo = np.cos(np.radians(bin_edges[:-1]))
    cos_hi = np.cos(np.radians(bin_edges[1:]))
    solid_angle = 2.0 * np.pi * np.abs(cos_lo - cos_hi)

    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = k_center * counts / (n_events * dk * solid_angle)
        rel_err = np.where(counts > 0, 1.0 / np.sqrt(counts), 0.0)

    return theta_centers, intensity, rel_err


# ---------------------------------------------------------------------------
# Our semi-analytical calculation
# ---------------------------------------------------------------------------
def calc_spectrum(angle_deg: float, k_values: np.ndarray) -> np.ndarray:
    """Compute our thick-target intensity at given k values and angle."""
    from server.physics.thick_target import thick_target_intensity

    mat_sym = MAT if MAT in ("Fe",) else None
    intensities = np.array([
        thick_target_intensity(T0, float(k), angle_deg, Z, A, RHO, mat_sym)
        for k in k_values
    ])
    return intensities


def calc_angular(k_mev: float, angles: np.ndarray) -> np.ndarray:
    """Compute our thick-target intensity at given angles for fixed k."""
    from server.physics.thick_target import thick_target_intensity

    mat_sym = MAT if MAT in ("Fe",) else None
    return np.array([
        thick_target_intensity(T0, k_mev, float(a), Z, A, RHO, mat_sym)
        for a in angles
    ])


# ---------------------------------------------------------------------------
# NASA data (if available)
# ---------------------------------------------------------------------------
def get_nasa(angle_deg: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Get NASA grid data for Fe at 3 MeV if available."""
    try:
        from server.physics.interpolation import get_nasa_spectrum_at_grid_point
        k, i = get_nasa_spectrum_at_grid_point(T0, angle_deg, MAT)
        if k:
            return np.array(k), np.array(i)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def plot_spectrum_at_angle(
    photons: np.ndarray, n_events: int, angle_deg: float, save: bool = True
) -> None:
    """Plot energy spectrum at a specific detection angle."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Geant4 histogram
    k_g4, i_g4, err_g4 = bin_spectrum(photons, angle_deg, 5.0, n_events)
    nonzero = i_g4 > 0
    ax.errorbar(
        k_g4[nonzero], i_g4[nonzero],
        yerr=i_g4[nonzero] * err_g4[nonzero],
        fmt="o-", color=C_G4, markersize=4, linewidth=1.5, capsize=2,
        label=f"Geant4 MC ({n_events/1e6:.0f}M e-)",
    )

    # Our calculation
    i_calc = calc_spectrum(angle_deg, k_g4)
    ax.plot(k_g4, i_calc, "-", color=C_CALC, linewidth=2, label="Semi-analytical (BremsLib)")

    # NASA data
    nasa = get_nasa(angle_deg)
    if nasa is not None:
        k_n, i_n = nasa
        ax.plot(k_n, i_n, "s--", color=C_NASA, markersize=5, linewidth=1.5,
                label="NASA TN D-4755")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Photon Energy (MeV)")
    ax.set_ylabel("Intensity  (MeV / MeV sr electron)")
    ax.set_title(f"Fe, {T0} MeV e-, {angle_deg:.0f} deg")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.02, 3.0)

    if save:
        out = OUT_DIR / f"{PREFIX}_spectrum_{angle_deg:.0f}deg.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out.name}")
    plt.close(fig)


def plot_all_angles(photons: np.ndarray, n_events: int) -> None:
    """6-panel figure: spectra at 0, 30, 60, 90, 120, 150 deg."""
    angles = [0, 30, 60, 90, 120, 150]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)

    for ax_obj, angle in zip(axes.flat, angles):
        k_g4, i_g4, err_g4 = bin_spectrum(photons, angle, 5.0, n_events)
        nonzero = i_g4 > 0
        ax_obj.errorbar(
            k_g4[nonzero], i_g4[nonzero],
            yerr=i_g4[nonzero] * err_g4[nonzero],
            fmt="o-", color=C_G4, markersize=3, linewidth=1, capsize=1,
            label="Geant4",
        )
        i_calc = calc_spectrum(angle, k_g4)
        ax_obj.plot(k_g4, i_calc, "-", color=C_CALC, linewidth=1.5, label="Calc")

        nasa = get_nasa(float(angle))
        if nasa is not None:
            ax_obj.plot(nasa[0], nasa[1], "s--", color=C_NASA, markersize=4, label="NASA")

        ax_obj.set_xscale("log")
        ax_obj.set_yscale("log")
        ax_obj.set_title(f"{angle} deg", fontsize=12)
        ax_obj.grid(True, which="both", alpha=0.3)
        ax_obj.set_xlim(0.03, 2.9)
        if angle == 0:
            ax_obj.legend(fontsize=8)

    fig.suptitle(f"Fe, {T0} MeV electrons — Bremsstrahlung Spectra", fontsize=15, y=0.98)
    fig.supxlabel("Photon Energy (MeV)", fontsize=12)
    fig.supylabel("Intensity (MeV / MeV sr electron)", fontsize=12)
    fig.tight_layout()

    out = OUT_DIR / f"{PREFIX}_all_angles.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out.name}")
    plt.close(fig)


def plot_angular_distribution(photons: np.ndarray, n_events: int) -> None:
    """Angular distribution at several photon energies."""
    fig, ax = plt.subplots(figsize=(10, 7))

    energies = [0.1, 0.3, 0.5, 1.0, 2.0]
    colors = ["#ff6b6b", "#ffa07a", "#ffd700", "#98fb98", "#87ceeb"]

    for k_mev, color in zip(energies, colors):
        hw = k_mev * 0.15  # +/- 15% energy window
        th, i_g4, err = angular_distribution(photons, k_mev, hw, n_events)
        nonzero = i_g4 > 0
        ax.errorbar(
            th[nonzero], i_g4[nonzero],
            yerr=i_g4[nonzero] * err[nonzero],
            fmt="o-", color=color, markersize=3, linewidth=1.5, capsize=1,
            label=f"G4: k={k_mev} MeV",
        )

        # Our calculation at same angles
        i_calc = calc_angular(k_mev, th[nonzero])
        ax.plot(th[nonzero], i_calc, "--", color=color, linewidth=1, alpha=0.7)

    ax.set_yscale("log")
    ax.set_xlabel("Detection Angle (degrees)")
    ax.set_ylabel("Intensity (MeV / MeV sr electron)")
    ax.set_title(f"Fe, {T0} MeV e- — Angular Distribution")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 180)

    out = OUT_DIR / f"{PREFIX}_angular.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out.name}")
    plt.close(fig)


def plot_electron_distribution(electrons: np.ndarray, n_events: int) -> None:
    """Scattered electron angular and energy distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Angular distribution
    bin_edges = np.linspace(0, 180, 37)
    counts, _ = np.histogram(electrons["theta"], bins=bin_edges)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cos_lo = np.cos(np.radians(bin_edges[:-1]))
    cos_hi = np.cos(np.radians(bin_edges[1:]))
    solid = 2.0 * np.pi * np.abs(cos_lo - cos_hi)
    rate = counts / (n_events * solid)

    ax1.bar(centers, rate, width=4.5, color=C_ELEC, alpha=0.7, edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Angle from beam (degrees)")
    ax1.set_ylabel("Electrons / (sr electron)")
    ax1.set_title("Escaped Electron Angular Distribution")
    ax1.grid(True, alpha=0.3)

    # Energy spectrum
    k_edges = np.linspace(0, T0, 50)
    counts_e, _ = np.histogram(electrons["k"], bins=k_edges)
    k_cen = 0.5 * (k_edges[:-1] + k_edges[1:])
    dk = np.diff(k_edges)
    rate_e = counts_e / (n_events * dk)

    ax2.bar(k_cen, rate_e, width=dk[0] * 0.9, color=C_ELEC, alpha=0.7,
            edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Electron Energy (MeV)")
    ax2.set_ylabel("Electrons / (MeV electron)")
    ax2.set_title("Escaped Electron Energy Spectrum")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Fe, {T0} MeV — Scattered Electrons at Kill Sphere", fontsize=14)
    fig.tight_layout()

    out = OUT_DIR / f"{PREFIX}_electron_dist.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out.name}")
    plt.close(fig)


def plot_energy_balance(photons: np.ndarray, electrons: np.ndarray, n_events: int) -> None:
    """Energy balance: deposited vs photons vs electrons."""
    e_photons = float(np.sum(photons["k"])) / n_events
    e_electrons = float(np.sum(electrons["k"])) / n_events
    e_deposited = T0 - e_photons - e_electrons

    fig, ax = plt.subplots(figsize=(8, 8))
    labels = [
        f"Deposited\n{e_deposited:.3f} MeV",
        f"Photons\n{e_photons:.3f} MeV",
        f"Electrons\n{e_electrons:.3f} MeV",
    ]
    sizes = [e_deposited, e_photons, e_electrons]
    colors_pie = ["#4a86c8", C_G4, C_ELEC]
    explode = (0.02, 0.05, 0.05)

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_pie, explode=explode,
        autopct="%1.1f%%", startangle=90, textprops={"color": "white", "fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")

    ax.set_title(f"Energy Balance: Fe, {T0} MeV/electron\n({n_events/1e6:.0f}M events)",
                 fontsize=14)

    out = OUT_DIR / f"{PREFIX}_energy_balance.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out.name}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"=== Geant4 Results Plotter: {PREFIX} ===\n")

    if not PHOTON_FILE.exists():
        print(f"Photon file not found: {PHOTON_FILE}")
        print("Run the simulation first or check the prefix.")
        sys.exit(1)

    n_events = get_n_events(PHOTON_FILE)
    photons = load_photons(PHOTON_FILE)

    electrons = None
    if ELECTRON_FILE.exists():
        electrons = load_electrons(ELECTRON_FILE)

    if SUMMARY_FILE.exists():
        print(f"\n{SUMMARY_FILE.read_text()}")

    print("\nGenerating plots...")

    # Energy spectra at key angles
    for angle in [0, 30, 60, 90]:
        plot_spectrum_at_angle(photons, n_events, angle)

    # Multi-panel all angles
    plot_all_angles(photons, n_events)

    # Angular distribution
    plot_angular_distribution(photons, n_events)

    # Electron distributions
    if electrons is not None and len(electrons) > 0:
        plot_electron_distribution(electrons, n_events)
        plot_energy_balance(photons, electrons, n_events)

    print("\nDone!")


if __name__ == "__main__":
    main()
