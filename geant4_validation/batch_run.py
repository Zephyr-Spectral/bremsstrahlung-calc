#!/usr/bin/env python3
"""Batch Geant4 thick-target bremsstrahlung simulation.

Runs the kill-sphere simulation for 11 materials (NASA 10 + graphite)
at 0.5-10 MeV in 0.5 MeV steps, 10M electrons each.

After completion, post-processes all binary files into a single
compressed lookup table (geant4_lookup.npz).

Usage:
    python batch_run.py              # run everything
    python batch_run.py --dry-run    # show what would be done
    python batch_run.py --resume     # skip already-completed runs
    python batch_run.py --build-only # only build lookup table from existing data

Estimated runtime: ~6 hours on Apple M3 Max.
Output: ~6 GB raw binary + ~50 MB compressed lookup table.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
G4_EXE = SCRIPT_DIR / "fe_3mev_killsphere"  # same binary, takes material as arg
G4_ENV = SCRIPT_DIR / "g4env.sh"
OUTPUT_DIR = SCRIPT_DIR / "batch_data"
LOOKUP_FILE = SCRIPT_DIR / "geant4_lookup.npz"

N_EVENTS = 10_000_000

# Materials: NASA 10 + graphite (C)
MATERIALS = {
    "C":  {"Z": 6,  "A": 12.011, "density": 2.267, "g4_name": "G4_C",  "name": "Graphite"},
    "Mg": {"Z": 12, "A": 24.305, "density": 1.738, "g4_name": "G4_Mg", "name": "Magnesium"},
    "Al": {"Z": 13, "A": 26.982, "density": 2.699, "g4_name": "G4_Al", "name": "Aluminum"},
    "Ti": {"Z": 22, "A": 47.867, "density": 4.507, "g4_name": "G4_Ti", "name": "Titanium"},
    "Mn": {"Z": 25, "A": 54.938, "density": 7.470, "g4_name": "G4_Mn", "name": "Manganese"},
    "Fe": {"Z": 26, "A": 55.845, "density": 7.874, "g4_name": "G4_Fe", "name": "Iron"},
    "Ni": {"Z": 28, "A": 58.693, "density": 8.908, "g4_name": "G4_Ni", "name": "Nickel"},
    "Cu": {"Z": 29, "A": 63.546, "density": 8.960, "g4_name": "G4_Cu", "name": "Copper"},
    "W":  {"Z": 74, "A": 183.84, "density": 19.25, "g4_name": "G4_W",  "name": "Tungsten"},
    "Au": {"Z": 79, "A": 196.97, "density": 19.30, "g4_name": "G4_Au", "name": "Gold"},
    "Pb": {"Z": 82, "A": 207.2,  "density": 11.34, "g4_name": "G4_Pb", "name": "Lead"},
}

# Energy grid: 0.5 to 10.0 MeV in 0.5 MeV steps
ENERGIES_MEV = [round(0.5 * i, 1) for i in range(1, 21)]

# Binning for lookup table
THETA_BINS = np.linspace(0, 180, 37)      # 36 bins, 5 deg each
K_BINS_N = 40                               # log-spaced energy bins per run
K_MIN_FRAC = 0.005                          # 0.5% of T0 or 10 keV, whichever is larger
K_MIN_ABS = 0.010                           # 10 keV absolute floor


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------
def progress_bar(current: int, total: int, prefix: str = "", width: int = 40) -> str:
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    return f"\r{prefix} |{bar}| {current}/{total} ({frac*100:.1f}%)"


def eta_string(elapsed_s: float, current: int, total: int) -> str:
    if current == 0:
        return "estimating..."
    rate = elapsed_s / current
    remaining = rate * (total - current)
    if remaining > 3600:
        return f"{remaining/3600:.1f}h remaining"
    if remaining > 60:
        return f"{remaining/60:.0f}m remaining"
    return f"{remaining:.0f}s remaining"


# ---------------------------------------------------------------------------
# CSDA range computation (standalone, no server imports needed)
# ---------------------------------------------------------------------------
def csda_range_simple(t0_mev: float, z: int, a: float) -> float:
    """Rough CSDA range in g/cm2 using Berger-Seltzer scaling.

    Good enough for setting target thickness (we add 1mm margin anyway).
    """
    # Use ESTAR data if available, otherwise approximate
    try:
        sys.path.insert(0, str(SCRIPT_DIR.parent))
        from server.physics.electron_range import csda_range
        return csda_range(t0_mev, z, a)
    except Exception:
        pass
    # Fallback: rough scaling R ~ (T0/MeV)^1.7 / (Z^0.3 * rho_eff)
    return 0.5 * t0_mev**1.7 / (z**0.3)


# ---------------------------------------------------------------------------
# Build the Geant4 executable
# ---------------------------------------------------------------------------
BUILD_SCRIPT: Path = SCRIPT_DIR / "build_killsphere.sh"


def ensure_executable() -> bool:
    """Check that the generalized killsphere executable exists, build if not."""
    gen_exe = SCRIPT_DIR / "killsphere_general"
    if gen_exe.exists():
        return True
    print("Building generalized kill sphere executable...")
    return build_general_executable()


def build_general_executable() -> bool:
    """Build the generalized executable via the build shell script.

    The build script (build_killsphere.sh) handles conda activation,
    geant4-config flag extraction, and compilation. No shell interpolation
    in Python — all arguments are passed as a safe list.
    """
    src = SCRIPT_DIR / "killsphere_general.cc"
    exe = SCRIPT_DIR / "killsphere_general"

    _write_general_source(src)

    if not BUILD_SCRIPT.exists():
        print(f"Build script not found: {BUILD_SCRIPT}")
        return False

    try:
        result = subprocess.run(
            [str(BUILD_SCRIPT), str(src), str(exe)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(SCRIPT_DIR),
        )
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr[:500]}")
            return False
        print(result.stdout.strip())
        return True
    except subprocess.TimeoutExpired:
        print("Build timed out after 120s")
        return False


def _write_general_source(path: Path) -> None:
    """Write generalized kill sphere C++ source that takes CLI args."""
    path.write_text(r'''
/// Generalized kill sphere: takes material, energy, thickness, n_events, prefix as args.
/// Usage: ./killsphere_general <G4_material> <T0_MeV> <thickness_mm> <target_R_mm> <n_events> <prefix>

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VSensitiveDetector.hh"
#include "G4SDManager.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Sphere.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleGun.hh"
#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4VModularPhysicsList.hh"
#include "G4DecayPhysics.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4TouchableHistory.hh"
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <mutex>
#include <atomic>

// CLI-configurable globals
static std::string g_material = "G4_Fe";
static double g_T0_MeV = 3.0;
static double g_target_z_mm = 3.6;
static double g_target_r_mm = 9.6;
static double g_beam_r_mm = 5.0;
static double g_sphere_r_mm = 200.0;
static int g_n_events = 10000000;
static std::string g_prefix = "output";

#pragma pack(push, 1)
struct PhotonRec  { float k, theta, phi; };
struct ElectronRec { float k, theta, phi, x, y, z; };
#pragma pack(pop)

static std::mutex g_ph_mtx, g_el_mtx, g_dose_mtx;
static std::ofstream g_ph_file, g_el_file;
static std::atomic<long> g_ph_count{0}, g_el_count{0}, g_bs_count{0};
static double g_dose = 0, g_dose_sq = 0;

class KillSphereSD : public G4VSensitiveDetector {
public:
    KillSphereSD(const G4String& n) : G4VSensitiveDetector(n) {}
    G4bool ProcessHits(G4Step* s, G4TouchableHistory*) override {
        auto* t = s->GetTrack();
        double k = t->GetKineticEnergy()/MeV;
        G4ThreeVector d = t->GetMomentumDirection();
        double th = std::acos(std::max(-1.0,std::min(1.0,d.z())))*180/M_PI;
        double ph = std::atan2(d.y(),d.x())*180/M_PI; if(ph<0) ph+=360;
        if (t->GetDefinition()==G4Gamma::Definition() && k>0.001) {
            PhotonRec r{(float)k,(float)th,(float)ph};
            { std::lock_guard<std::mutex> l(g_ph_mtx);
              g_ph_file.write((char*)&r,sizeof(r)); }
            g_ph_count++;
        } else if (t->GetDefinition()==G4Electron::Definition() && k>0.001) {
            auto p=s->GetPreStepPoint()->GetPosition();
            ElectronRec r{(float)k,(float)th,(float)ph,(float)(p.x()/mm),(float)(p.y()/mm),(float)(p.z()/mm)};
            { std::lock_guard<std::mutex> l(g_el_mtx);
              g_el_file.write((char*)&r,sizeof(r)); }
            g_el_count++;
        }
        t->SetTrackStatus(fStopAndKill);
        return true;
    }
    void Initialize(G4HCofThisEvent*) override {}
    void EndOfEvent(G4HCofThisEvent*) override {}
};

class StepAction : public G4UserSteppingAction {
public:
    void UserSteppingAction(const G4Step* s) override {
        auto* v=s->GetPreStepPoint()->GetPhysicalVolume();
        if(!v||v->GetName()!="Target") return;
        double e=s->GetTotalEnergyDeposit()/MeV;
        if(e>0){ std::lock_guard<std::mutex> l(g_dose_mtx); g_dose+=e; g_dose_sq+=e*e; }
        auto* t=s->GetTrack();
        auto* pv=s->GetPostStepPoint()->GetPhysicalVolume();
        if(t->GetDefinition()==G4Electron::Definition()&&pv&&pv->GetName()=="World")
            if(s->GetPostStepPoint()->GetPosition().z()<0.01*mm) g_bs_count++;
    }
};

class Det : public G4VUserDetectorConstruction {
public:
    G4VPhysicalVolume* Construct() override {
        auto* n=G4NistManager::Instance();
        auto* mat=n->FindOrBuildMaterial(g_material);
        auto* vac=n->FindOrBuildMaterial("G4_Galactic");
        double wh=(g_sphere_r_mm+50)*mm;
        auto* ws=new G4Box("World",wh,wh,wh);
        auto* wl=new G4LogicalVolume(ws,vac,"World");
        auto* wp=new G4PVPlacement(nullptr,{},wl,"World",nullptr,false,0);
        auto* ts=new G4Tubs("Target",0,g_target_r_mm*mm,g_target_z_mm/2*mm,0,360*deg);
        auto* tl=new G4LogicalVolume(ts,mat,"Target");
        new G4PVPlacement(nullptr,G4ThreeVector(0,0,g_target_z_mm/2*mm),tl,"Target",wl,false,0);
        auto* ss=new G4Sphere("KS",g_sphere_r_mm*mm,(g_sphere_r_mm+1)*mm,0,360*deg,0,180*deg);
        auto* sl=new G4LogicalVolume(ss,vac,"KillSphere");
        new G4PVPlacement(nullptr,G4ThreeVector(0,0,g_target_z_mm/2*mm),sl,"KillSphere",wl,false,0);
        return wp;
    }
    void ConstructSDandField() override {
        auto* sd=new KillSphereSD("KS_SD");
        G4SDManager::GetSDMpointer()->AddNewDetector(sd);
        SetSensitiveDetector("KillSphere",sd);
    }
};

class Gun : public G4VUserPrimaryGeneratorAction {
    G4ParticleGun* g;
public:
    Gun():g(new G4ParticleGun(1)){
        g->SetParticleDefinition(G4Electron::Definition());
        g->SetParticleMomentumDirection(G4ThreeVector(0,0,1));
        g->SetParticleEnergy(g_T0_MeV*MeV);
    }
    ~Gun() override{delete g;}
    void GeneratePrimaries(G4Event* e) override {
        double r=g_beam_r_mm*mm*std::sqrt(G4UniformRand());
        double p=2*M_PI*G4UniformRand();
        g->SetParticlePosition(G4ThreeVector(r*std::cos(p),r*std::sin(p),-5*mm));
        g->GeneratePrimaryVertex(e);
    }
};

class Phys : public G4VModularPhysicsList {
public:
    Phys(){ RegisterPhysics(new G4EmStandardPhysics_option4()); RegisterPhysics(new G4DecayPhysics()); }
};

int main(int argc, char** argv) {
    if(argc<7){
        std::cerr<<"Usage: "<<argv[0]<<" <G4_mat> <T0_MeV> <thick_mm> <radius_mm> <n_events> <prefix>"<<std::endl;
        return 1;
    }
    g_material=argv[1]; g_T0_MeV=std::stod(argv[2]); g_target_z_mm=std::stod(argv[3]);
    g_target_r_mm=std::stod(argv[4]); g_n_events=std::stoi(argv[5]); g_prefix=argv[6];

    g_ph_file.open(g_prefix+"_photons.bin",std::ios::binary);
    g_el_file.open(g_prefix+"_electrons.bin",std::ios::binary);
    int h1[3]={g_n_events,(int)sizeof(PhotonRec),2};
    int h2[3]={g_n_events,(int)sizeof(ElectronRec),2};
    g_ph_file.write((char*)h1,12); g_el_file.write((char*)h2,12);

    auto* rm=G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
    rm->SetUserInitialization(new Det()); rm->SetUserInitialization(new Phys());
    rm->SetUserAction(new Gun()); rm->SetUserAction(new StepAction());
    rm->Initialize();
    G4UImanager::GetUIpointer()->ApplyCommand("/run/verbose 0");
    G4UImanager::GetUIpointer()->ApplyCommand("/event/verbose 0");
    G4UImanager::GetUIpointer()->ApplyCommand("/tracking/verbose 0");
    G4UImanager::GetUIpointer()->ApplyCommand("/run/printProgress 1000000");

    std::cerr<<g_material<<" "<<g_T0_MeV<<" MeV, "<<g_target_z_mm<<" mm, "
             <<g_n_events<<" events -> "<<g_prefix<<std::endl;
    rm->BeamOn(g_n_events);
    g_ph_file.close(); g_el_file.close();

    long np=g_ph_count, ne=g_el_count, nb=g_bs_count;
    double dm=g_dose/g_n_events;

    // Write summary JSON with physics model + data library info
    std::string g4ver = "11.3.2";  // from geant4-config --version
    const char* emlow = std::getenv("G4LEDATA");
    const char* ensdf = std::getenv("G4ENSDFSTATEDATA");
    std::ofstream sf(g_prefix+"_summary.json");
    sf<<"{"
      <<"\"material\":\""<<g_material<<"\","
      <<"\"T0_MeV\":"<<g_T0_MeV<<","
      <<"\"target_z_mm\":"<<g_target_z_mm<<","
      <<"\"target_r_mm\":"<<g_target_r_mm<<","
      <<"\"beam_r_mm\":"<<g_beam_r_mm<<","
      <<"\"sphere_r_mm\":"<<g_sphere_r_mm<<","
      <<"\"n_events\":"<<g_n_events<<","
      <<"\"n_photons\":"<<np<<","
      <<"\"n_electrons\":"<<ne<<","
      <<"\"backscatter_pct\":"<<100.0*nb/g_n_events<<","
      <<"\"dose_MeV_per_e\":"<<dm<<","
      <<"\"geant4_version\":\""<<g4ver<<"\","
      <<"\"physics_list\":\"G4EmStandardPhysics_option4\","
      <<"\"brems_model\":\"eBremSB (Seltzer-Berger)\","
      <<"\"brems_angular\":\"AngularGen2BS\","
      <<"\"msc_model\":\"GoudsmitSaunderson + WentzelVIUni\","
      <<"\"ionisation_model\":\"PenIoni + MollerBhabha\","
      <<"\"G4LEDATA\":\""<<(emlow?emlow:"unknown")<<"\","
      <<"\"G4ENSDFSTATEDATA\":\""<<(ensdf?ensdf:"unknown")<<"\""
      <<"}"<<std::endl;
    sf.close();

    std::cerr<<"  photons="<<np<<" electrons="<<ne<<" bs="<<100.0*nb/g_n_events
             <<"% dose="<<dm<<" MeV/e"<<std::endl;

    delete rm;
    return 0;
}
''')


# ---------------------------------------------------------------------------
# Run a single simulation
# ---------------------------------------------------------------------------
def run_single(
    mat_symbol: str, t0_mev: float, dry_run: bool = False
) -> dict | None:
    """Run one Geant4 simulation. Returns summary dict or None on failure."""
    mat = MATERIALS[mat_symbol]
    g4_name = mat["g4_name"]
    z, a, rho = mat["Z"], mat["A"], mat["density"]

    # Compute target dimensions
    r_csda = csda_range_simple(t0_mev, z, a)
    thickness_mm = r_csda / rho * 10 + 1.0  # CSDA range in mm + 1mm margin
    radius_mm = 5.0 + thickness_mm + 2.0     # beam radius + range + margin

    prefix = str(OUTPUT_DIR / f"{mat_symbol}_{t0_mev:.1f}MeV")

    if dry_run:
        print(f"  {mat_symbol:2s} {t0_mev:5.1f} MeV: thick={thickness_mm:.2f}mm "
              f"R={radius_mm:.1f}mm -> {prefix}")
        return None

    exe = SCRIPT_DIR / "killsphere_general"
    cmd = [
        str(G4_ENV), str(exe),
        g4_name, str(t0_mev), f"{thickness_mm:.3f}", f"{radius_mm:.1f}",
        str(N_EVENTS), prefix,
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(SCRIPT_DIR),
        )
    except subprocess.TimeoutExpired:
        print(f"\n  TIMEOUT: {mat_symbol} {t0_mev} MeV")
        return None

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED: {mat_symbol} {t0_mev} MeV (exit {result.returncode})")
        return None

    # Read summary
    summary_path = Path(prefix + "_summary.json")
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)
        summary["elapsed_s"] = round(elapsed, 1)
        summary["symbol"] = mat_symbol
        return summary

    return {"symbol": mat_symbol, "T0_MeV": t0_mev, "elapsed_s": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# Build lookup table from all binary files
# ---------------------------------------------------------------------------
def build_lookup_table() -> None:
    """Post-process all binary files into a single compressed .npz lookup."""
    print("\nBuilding lookup table...")

    n_mat = len(MATERIALS)
    n_e = len(ENERGIES_MEV)
    n_theta = len(THETA_BINS) - 1   # 36 angular bins
    n_k = K_BINS_N

    mat_list = list(MATERIALS.keys())

    # We'll store: intensity[mat, energy, theta_bin, k_bin]
    # Plus the bin edges for reconstruction
    all_intensity = np.zeros((n_mat, n_e, n_theta, n_k), dtype=np.float64)
    all_counts = np.zeros((n_mat, n_e, n_theta, n_k), dtype=np.int64)
    summaries = []

    for i_m, mat_sym in enumerate(mat_list):
        for i_e, t0 in enumerate(ENERGIES_MEV):
            prefix = OUTPUT_DIR / f"{mat_sym}_{t0:.1f}MeV"
            ph_path = Path(str(prefix) + "_photons.bin")

            if not ph_path.exists():
                continue

            # Read photons
            with ph_path.open("rb") as f:
                header = struct.unpack("3i", f.read(12))
                n_events = header[0]
                dt = np.dtype([("k", "f4"), ("theta", "f4"), ("phi", "f4")])
                photons = np.fromfile(f, dtype=dt)

            # Energy bins for this T0
            k_min = max(K_MIN_ABS, K_MIN_FRAC * t0)
            k_max = 0.95 * t0
            k_edges = np.logspace(np.log10(k_min), np.log10(k_max), n_k + 1)
            k_centers = np.sqrt(k_edges[:-1] * k_edges[1:])
            dk = np.diff(k_edges)

            # Bin into (theta, k) grid
            for i_th in range(n_theta):
                th_lo, th_hi = THETA_BINS[i_th], THETA_BINS[i_th + 1]
                cos_lo = np.cos(np.radians(th_lo))
                cos_hi = np.cos(np.radians(th_hi))
                solid_angle = 2 * np.pi * abs(cos_lo - cos_hi)

                mask = (photons["theta"] >= th_lo) & (photons["theta"] < th_hi)
                k_sel = photons["k"][mask]

                counts, _ = np.histogram(k_sel, bins=k_edges)
                all_counts[i_m, i_e, i_th, :] = counts

                # Intensity: I = k * N / (N_elec * dk * Omega)
                with np.errstate(divide="ignore", invalid="ignore"):
                    intensity = k_centers * counts / (n_events * dk * solid_angle)
                all_intensity[i_m, i_e, i_th, :] = intensity

            # Read summary
            sum_path = Path(str(prefix) + "_summary.json")
            if sum_path.exists():
                with sum_path.open() as f:
                    summaries.append(json.load(f))

        print(f"  Processed {mat_sym}")

    # Save
    np.savez_compressed(
        LOOKUP_FILE,
        intensity=all_intensity,
        counts=all_counts,
        materials=np.array(mat_list),
        energies_mev=np.array(ENERGIES_MEV),
        theta_bin_edges=THETA_BINS,
        n_events=N_EVENTS,
    )

    size_mb = LOOKUP_FILE.stat().st_size / 1024 / 1024
    print(f"Saved {LOOKUP_FILE.name} ({size_mb:.1f} MB)")
    print(f"Shape: {all_intensity.shape} (materials x energies x theta x k)")

    # Save summaries
    summary_file = SCRIPT_DIR / "batch_summaries.json"
    with summary_file.open("w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved {summary_file.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    dry_run = "--dry-run" in sys.argv
    resume = "--resume" in sys.argv
    build_only = "--build-only" in sys.argv

    if build_only:
        build_lookup_table()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the generalized executable
    if not dry_run:
        if not ensure_executable():
            print("Failed to build executable. Exiting.")
            sys.exit(1)

    # Plan all runs
    runs = []
    for mat_sym in MATERIALS:
        for t0 in ENERGIES_MEV:
            prefix = OUTPUT_DIR / f"{mat_sym}_{t0:.1f}MeV"
            done = Path(str(prefix) + "_summary.json").exists()
            if resume and done:
                continue
            runs.append((mat_sym, t0))

    total = len(runs)
    print(f"{'[DRY RUN] ' if dry_run else ''}Batch Geant4 Simulation")
    print(f"Materials: {len(MATERIALS)}")
    print(f"Energies: {len(ENERGIES_MEV)} ({ENERGIES_MEV[0]}-{ENERGIES_MEV[-1]} MeV)")
    print(f"Events per run: {N_EVENTS:,}")
    print(f"Total runs: {total}")
    if resume:
        skipped = len(MATERIALS) * len(ENERGIES_MEV) - total
        print(f"Skipped (already done): {skipped}")
    print()

    if dry_run:
        for mat_sym, t0 in runs:
            run_single(mat_sym, t0, dry_run=True)
        return

    # Execute all runs
    t_start = time.time()
    completed = 0
    failed = 0

    for i, (mat_sym, t0) in enumerate(runs):
        sys.stderr.write(progress_bar(i, total, "Running") +
                         f"  {mat_sym} {t0:.1f} MeV  " +
                         eta_string(time.time() - t_start, i, total) + "  ")
        sys.stderr.flush()

        summary = run_single(mat_sym, t0)
        if summary is not None:
            completed += 1
            np_str = f"{summary.get('n_photons', '?')}"
            sys.stderr.write(f"({np_str} photons, {summary.get('elapsed_s', '?')}s)\n")
        else:
            failed += 1
            sys.stderr.write("FAILED\n")

    sys.stderr.write(progress_bar(total, total, "Running") + "  Done!\n")
    elapsed_total = time.time() - t_start
    print(f"\nCompleted: {completed}, Failed: {failed}")
    print(f"Total time: {elapsed_total/3600:.1f} hours")

    # Build lookup table
    build_lookup_table()


if __name__ == "__main__":
    main()
