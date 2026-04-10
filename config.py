"""Central configuration — single source of truth for all constants, paths, and material data."""

from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
NASA_DATA_PATH: Path = DATA_DIR / "nasa_tnd4755" / "tables.json"
XCOM_DATA_PATH: Path = DATA_DIR / "xcom_elements.json"
ESTAR_DATA_PATH: Path = DATA_DIR / "estar_elements.json"
SB_DATA_PATH: Path = DATA_DIR / "sb_elements.json"  # Seltzer-Berger scaled DCS tables
STATIC_DIR: Path = PROJECT_ROOT / "static"
TEMPLATES_DIR: Path = PROJECT_ROOT / "templates"

# Monte Carlo
GEANT4_EXE: Path = PROJECT_ROOT / "geant4_validation" / "thick_target_brems"
GEANT4_ENV_SCRIPT: Path = PROJECT_ROOT / "geant4_validation" / "g4env.sh"
EGSNRC_HOME: Path = Path.home() / "Projects" / "EGSnrc"
MC_CACHE_DIR: Path = DATA_DIR / "mc_cache"
MC_DEFAULT_EVENTS: int = 100_000
MC_TIMEOUT_SECONDS: int = 300

# Geant4 NIST material names
G4_MATERIAL_NAMES: dict[str, str] = {
    "Mg": "G4_Mg",
    "Al": "G4_Al",
    "Ti": "G4_Ti",
    "Mn": "G4_Mn",
    "Fe": "G4_Fe",
    "Ni": "G4_Ni",
    "Cu": "G4_Cu",
    "W": "G4_W",
    "Au": "G4_Au",
    "Pb": "G4_Pb",
}

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_HOST: str = "127.0.0.1"
SERVER_PORT: int = 8001  # avoids chart-of-nuclides on 8000

# ---------------------------------------------------------------------------
# Physical constants (SI-compatible, energies in MeV where noted)
# ---------------------------------------------------------------------------
ELECTRON_MASS_MEV: float = 0.510998950  # MeV/c²
ELECTRON_MASS_KG: float = 9.10938e-31  # kg
CLASSICAL_ELECTRON_RADIUS_CM: float = 2.8179403e-13  # cm
FINE_STRUCTURE_CONSTANT: float = 1.0 / 137.035999
AVOGADRO: float = 6.02214076e23  # mol⁻¹
ELEMENTARY_CHARGE_C: float = 1.602176634e-19  # coulombs
SPEED_OF_LIGHT_CM_S: float = 2.99792458e10  # cm/s
PLANCK_CONSTANT_MEV_S: float = 4.135667696e-21  # MeV·s
BARN_CM2: float = 1.0e-24  # cm²

# Derived
RE_SQUARED_CM2: float = CLASSICAL_ELECTRON_RADIUS_CM**2  # r₀² in cm²
ALPHA_FINE: float = FINE_STRUCTURE_CONSTANT  # shorthand

# ---------------------------------------------------------------------------
# Material data type
# ---------------------------------------------------------------------------
type MaterialProperties = dict[str, int | float | str]

# ---------------------------------------------------------------------------
# NASA TN D-4755 materials (Table IV and throughout)
# Z: atomic number, A: atomic weight (g/mol), density: g/cm³, name: full name
# ---------------------------------------------------------------------------
NASA_MATERIALS: dict[str, MaterialProperties] = {
    "Mg": {"Z": 12, "A": 24.305, "density": 1.738, "name": "Magnesium"},
    "Al": {"Z": 13, "A": 26.982, "density": 2.699, "name": "Aluminum"},
    "Ti": {"Z": 22, "A": 47.867, "density": 4.507, "name": "Titanium"},
    "Mn": {"Z": 25, "A": 54.938, "density": 7.470, "name": "Manganese"},
    "Fe": {"Z": 26, "A": 55.845, "density": 7.874, "name": "Iron"},
    "Ni": {"Z": 28, "A": 58.693, "density": 8.908, "name": "Nickel"},
    "Cu": {"Z": 29, "A": 63.546, "density": 8.960, "name": "Copper"},
    "W": {"Z": 74, "A": 183.84, "density": 19.25, "name": "Tungsten"},
    "Au": {"Z": 79, "A": 196.97, "density": 19.30, "name": "Gold"},
    "Pb": {"Z": 82, "A": 207.2, "density": 11.34, "name": "Lead"},
}

# ---------------------------------------------------------------------------
# Extended materials (effective-Z composites)
# SS304/316: weighted by composition — effective Z ≈ 25.8 / 25.5
# ---------------------------------------------------------------------------
EXTENDED_MATERIALS: dict[str, MaterialProperties] = {
    "SS304": {
        "Z": 25.8,
        "A": 55.9,
        "density": 8.00,
        "name": "Stainless Steel 304",
    },
    "SS316": {
        "Z": 25.5,
        "A": 55.6,
        "density": 8.00,
        "name": "Stainless Steel 316",
    },
}

# Combined lookup
ALL_MATERIALS: dict[str, MaterialProperties] = {**NASA_MATERIALS, **EXTENDED_MATERIALS}

# XCOM element name lookup: material symbol -> name as stored in xcom_elements.json
XCOM_ELEMENT_NAMES: dict[str, str] = {
    "Mg": "Magnesium",
    "Al": "Aluminum",
    "Ti": "Titanium",
    "Mn": "Manganese",
    "Fe": "Iron",
    "Ni": "Nickel",
    "Cu": "Copper",
    "W": "Tungsten",
    "Au": "Gold",
    "Pb": "Lead",
}

# ---------------------------------------------------------------------------
# NASA TN D-4755 grid parameters
# ---------------------------------------------------------------------------
NASA_ELECTRON_ENERGIES_MEV: list[float] = [0.50, 0.75, 1.00, 2.00, 3.00]
NASA_DETECTION_ANGLES_DEG: list[float] = [0.0, 30.0, 60.0]

# ---------------------------------------------------------------------------
# Calculation defaults
# ---------------------------------------------------------------------------
DEFAULT_N_SLABS: int = 100  # depth slabs for thick-target integration
DEFAULT_PHOTON_ENERGY_POINTS: int = 200  # points in photon energy spectrum
DEFAULT_N_LEGENDRE: int = 20  # Legendre terms for scattering kernel
ENERGY_DEPTH_N_INTEGRATION: int = 500  # steps for T(m) energy-depth profile

# Energy range limits (MeV)
MIN_ELECTRON_ENERGY_MEV: float = 0.1
MAX_ELECTRON_ENERGY_MEV: float = 10.0

# Integration cutoffs and spectrum fraction bounds
CSDA_RANGE_EMIN_MEV: float = 0.01  # lower cutoff for CSDA range integration
SPECTRUM_K_FRACTION_MIN: float = 0.01  # k_min = fraction * T0 (thin-target)
SPECTRUM_K_FRACTION_MAX: float = 0.99  # k_max = fraction * T0 (thin-target)
THICK_SPECTRUM_K_FRACTION_MIN: float = 0.01  # k_min = fraction * T0 (thick-target)
THICK_SPECTRUM_K_MIN_MEV: float = 0.010  # absolute floor: 10 keV
THICK_SPECTRUM_K_FRACTION_MAX: float = 0.95  # k_max = fraction * T0 (thick-target)

# Quadrature defaults for angular integration
DEFAULT_THIN_TARGET_N_ANGLES: int = 32  # angle quadrature for thin-target integration
DEFAULT_THICK_TARGET_N_XI: int = 180  # electron angle points (~1 deg resolution)
DEFAULT_THICK_TARGET_N_AZIMUTH: int = 36  # azimuthal points (10 deg resolution)

# Scattering model parameters
MIN_COSINE_SLANT: float = 0.01  # floor for cos(phi_d) at near-grazing incidence

# ---------------------------------------------------------------------------
# Mean ionization potential parameterization (Bloch, eV)
# I ≈ 9.76*Z + 58.8*Z^(-0.19) for Z >= 13, else tabulated
# ---------------------------------------------------------------------------
MEAN_IONIZATION_POTENTIALS_EV: dict[int, float] = {
    1: 19.2,
    2: 41.8,
    3: 40.0,
    4: 63.7,
    5: 76.0,
    6: 78.0,
    7: 82.0,
    8: 95.0,
    9: 115.0,
    10: 137.0,
    11: 149.0,
    12: 156.0,
}


def mean_ionization_potential_ev(z: int | float) -> float:
    """Return mean ionization potential I in eV for atomic number Z.

    Uses tabulated values for Z <= 12, Bloch parameterization for Z >= 13.
    """
    z_int = round(z)
    if z_int <= 0:
        msg = f"Atomic number must be positive, got Z={z}"
        raise ValueError(msg)
    if z_int in MEAN_IONIZATION_POTENTIALS_EV:
        return float(MEAN_IONIZATION_POTENTIALS_EV[z_int])
    return float(9.76 * z_int + 58.8 * z_int ** (-0.19))


# ---------------------------------------------------------------------------
# Electron backscatter fraction: Cohen & Koral (NASA TN D-2909, 1965)
# Validated against Tabata (1971), Wright & Trump (1962), Powell (1968)
# ---------------------------------------------------------------------------
def backscatter_fraction(z: int | float, electron_energy_mev: float) -> float:
    """Number fraction of electrons backscattered from a thick target.

    Uses the Cohen & Koral (1965) empirical formula validated against 615
    experimental data points for Z >= 6, E = 0.05-22 MeV:

        eta = 1.28 * exp(-11.9 * Z^{-0.65} * (1 + 0.103 * Z^{0.37} * E^{0.65}))

    Validation:
        Pb 1.0 MeV: eta = 0.455  (Powell states ~0.44)
        Al 3.0 MeV: eta = 0.040  (Powell states ~0.04)

    References:
        Cohen JD & Koral KF, NASA TN D-2909 (1965)
        Tabata T, Ito R, Okabe S, NIM 94, 509 (1971)
    """
    import math

    if z <= 0:
        msg = f"Atomic number must be positive, got Z={z}"
        raise ValueError(msg)
    if electron_energy_mev <= 0:
        msg = f"Electron energy must be positive, got {electron_energy_mev} MeV"
        raise ValueError(msg)
    z_f = float(z)
    exponent = -11.9 * z_f ** (-0.65) * (1.0 + 0.103 * z_f**0.37 * electron_energy_mev**0.65)
    return float(min(1.28 * math.exp(exponent), 0.95))


# ---------------------------------------------------------------------------
# Utility: electron relativistic kinematics
# ---------------------------------------------------------------------------
def electron_gamma(kinetic_energy_mev: float) -> float:
    """Lorentz factor gamma = 1 + T / (m0 * c^2)."""
    return 1.0 + kinetic_energy_mev / ELECTRON_MASS_MEV


def electron_beta(kinetic_energy_mev: float) -> float:
    """Electron velocity β = v/c from kinetic energy."""
    gamma = electron_gamma(kinetic_energy_mev)
    return math.sqrt(1.0 - 1.0 / gamma**2)


def electron_momentum_moc(kinetic_energy_mev: float) -> float:
    """Electron momentum in units of m₀c."""
    total_energy = kinetic_energy_mev + ELECTRON_MASS_MEV
    return math.sqrt(total_energy**2 - ELECTRON_MASS_MEV**2) / ELECTRON_MASS_MEV


def electron_total_energy_mev(kinetic_energy_mev: float) -> float:
    """Total electron energy E = T + m₀c² in MeV."""
    return kinetic_energy_mev + ELECTRON_MASS_MEV
