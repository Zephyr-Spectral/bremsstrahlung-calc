/// Geant4 thick-target bremsstrahlung: Fe 3 MeV
/// Clean implementation: no detector geometry. Score photons at target boundary.
///
/// Geometry:
///   - Fe cylinder: 3.6 mm thick, 9.6 mm radius (CSDA range + margin)
///   - Beam: 1 cm diameter, 3 MeV electrons along +z, enters front face at z=0
///   - No detector volumes — photons scored when they leave the target
///
/// Scoring:
///   - Every photon leaving the target: energy, polar angle from +z (beam axis)
///   - Binned into 36 angular rings (5 deg each, 0-180) x 40 energy bins
///   - Energy deposited in target tracked for dose validation
///   - Electron backscatter counted
///
/// The polar angle theta is computed from the photon momentum direction,
/// NOT from its position. This gives the true emission angle relative to
/// the beam axis, which is what we want to compare with our calculation.
///
/// Usage: ./g4env.sh ./fe_3mev_tally [n_events]  (default 10000000)

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserSteppingAction.hh"

#include "G4Box.hh"
#include "G4Tubs.hh"
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

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <mutex>

// =====================================================================
// Configuration
// =====================================================================
static const int    N_RINGS   = 36;       // 5 deg per ring, 0-180
static const double D_THETA   = 5.0;      // degrees per ring
static const int    N_KBINS   = 40;       // energy bins (log-spaced)
static const double K_MIN     = 0.010;    // MeV
static const double K_MAX     = 3.000;    // MeV

// Fe target: CSDA range at 3 MeV = 2.59 mm, add 1 mm margin
static const double TARGET_Z  = 3.6;      // mm thickness
static const double TARGET_R  = 9.6;      // mm radius
static const double BEAM_R    = 5.0;      // mm (1 cm diameter)
static const double T0_MEV    = 3.0;

static int g_n_events = 10000000;

// =====================================================================
// Tallies
// =====================================================================
static std::mutex g_mutex;
static std::vector<std::vector<long>> g_counts(N_RINGS, std::vector<long>(N_KBINS, 0));
static double g_dose_mev = 0.0;
static long g_backscatter_count = 0;
static long g_total_photons = 0;

static std::vector<double> g_kbin_edges;

void init_bins() {
    g_kbin_edges.resize(N_KBINS + 1);
    double log_min = std::log10(K_MIN);
    double log_max = std::log10(K_MAX);
    for (int i = 0; i <= N_KBINS; i++) {
        g_kbin_edges[i] = std::pow(10.0, log_min + i * (log_max - log_min) / N_KBINS);
    }
}

int find_kbin(double k_mev) {
    if (k_mev < K_MIN || k_mev >= K_MAX) return -1;
    double log_k = std::log10(k_mev);
    double log_min = std::log10(K_MIN);
    double log_max = std::log10(K_MAX);
    int bin = (int)((log_k - log_min) / (log_max - log_min) * N_KBINS);
    return (bin < 0) ? 0 : (bin >= N_KBINS) ? N_KBINS - 1 : bin;
}

// =====================================================================
// Detector: just the Fe target cylinder in vacuum
// =====================================================================
class DetectorConstruction : public G4VUserDetectorConstruction {
public:
    G4VPhysicalVolume* Construct() override {
        auto* nist = G4NistManager::Instance();
        auto* fe = nist->FindOrBuildMaterial("G4_Fe");
        auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");

        // World: 50 cm cube (plenty of room)
        double world_half = 250.0 * mm;
        auto* world_solid = new G4Box("World", world_half, world_half, world_half);
        auto* world_log = new G4LogicalVolume(world_solid, vacuum, "World");
        auto* world_phys = new G4PVPlacement(nullptr, {}, world_log,
                                              "World", nullptr, false, 0);

        // Fe target: front face at z=0, back face at z=TARGET_Z
        auto* target_solid = new G4Tubs("Target", 0, TARGET_R * mm,
                                         TARGET_Z / 2 * mm, 0, 360 * deg);
        auto* target_log = new G4LogicalVolume(target_solid, fe, "Target");
        new G4PVPlacement(nullptr,
                          G4ThreeVector(0, 0, TARGET_Z / 2 * mm),
                          target_log, "Target", world_log, false, 0);

        return world_phys;
    }
};

// =====================================================================
// Beam: 1 cm diameter, 3 MeV electrons along +z
// =====================================================================
class PrimaryGenerator : public G4VUserPrimaryGeneratorAction {
    G4ParticleGun* fGun;

public:
    PrimaryGenerator() : fGun(new G4ParticleGun(1)) {
        fGun->SetParticleDefinition(G4Electron::Definition());
        fGun->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
        fGun->SetParticleEnergy(T0_MEV * MeV);
    }
    ~PrimaryGenerator() override { delete fGun; }

    void GeneratePrimaries(G4Event* evt) override {
        // Uniform random within 1 cm diameter circle
        double r = BEAM_R * mm * std::sqrt(G4UniformRand());
        double phi = 2.0 * M_PI * G4UniformRand();
        fGun->SetParticlePosition(
            G4ThreeVector(r * std::cos(phi), r * std::sin(phi), -0.1 * mm));
        fGun->GeneratePrimaryVertex(evt);
    }
};

// =====================================================================
// Stepping action: score at target boundary
// =====================================================================
class SteppingAction : public G4UserSteppingAction {
public:
    void UserSteppingAction(const G4Step* step) override {
        auto* track = step->GetTrack();
        auto* preVol = step->GetPreStepPoint()->GetPhysicalVolume();
        auto* postVol = step->GetPostStepPoint()->GetPhysicalVolume();
        if (!preVol || !postVol) return;

        const std::string& preName = preVol->GetName();
        const std::string& postName = postVol->GetName();

        // --- Dose in target ---
        if (preName == "Target") {
            double edep = step->GetTotalEnergyDeposit() / MeV;
            if (edep > 0) {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_dose_mev += edep;
            }
        }

        // --- Electron backscatter: electron leaving target through front face ---
        if (track->GetDefinition() == G4Electron::Definition() &&
            preName == "Target" && postName == "World") {
            if (step->GetPostStepPoint()->GetPosition().z() < 0.01 * mm) {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_backscatter_count++;
            }
        }

        // --- Photon leaving target into vacuum ---
        if (track->GetDefinition() != G4Gamma::Definition()) return;
        if (preName != "Target" || postName != "World") return;

        double k_mev = track->GetKineticEnergy() / MeV;
        if (k_mev < K_MIN) return;

        // Polar angle from beam axis (+z) using MOMENTUM DIRECTION
        G4ThreeVector dir = track->GetMomentumDirection();
        double cos_theta = dir.z();  // cos(angle from +z)
        double theta_deg = std::acos(std::max(-1.0, std::min(1.0, cos_theta)))
                           * 180.0 / M_PI;

        int ring = (int)(theta_deg / D_THETA);
        if (ring < 0 || ring >= N_RINGS) return;

        int kbin = find_kbin(k_mev);
        if (kbin < 0) return;

        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_counts[ring][kbin]++;
            g_total_photons++;
        }
    }
};

// =====================================================================
// Physics
// =====================================================================
class PhysicsList : public G4VModularPhysicsList {
public:
    PhysicsList() {
        RegisterPhysics(new G4EmStandardPhysics_option4());
        RegisterPhysics(new G4DecayPhysics());
    }
};

// =====================================================================
// Main
// =====================================================================
int main(int argc, char** argv) {
    if (argc > 1) g_n_events = std::stoi(argv[1]);
    init_bins();

    auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
    runManager->SetUserInitialization(new DetectorConstruction());
    runManager->SetUserInitialization(new PhysicsList());
    runManager->SetUserAction(new PrimaryGenerator());
    runManager->SetUserAction(new SteppingAction());
    runManager->Initialize();

    G4UImanager::GetUIpointer()->ApplyCommand("/run/verbose 0");
    G4UImanager::GetUIpointer()->ApplyCommand("/event/verbose 0");
    G4UImanager::GetUIpointer()->ApplyCommand("/tracking/verbose 0");
    G4UImanager::GetUIpointer()->ApplyCommand("/run/printProgress 1000000");

    std::cerr << "=== Fe 3 MeV Bremsstrahlung Tally ===" << std::endl;
    std::cerr << "Target: Fe cylinder " << TARGET_Z << " x " << TARGET_R
              << " mm (thickness x radius)" << std::endl;
    std::cerr << "Beam: " << 2 * BEAM_R << " mm dia, " << T0_MEV << " MeV e-" << std::endl;
    std::cerr << "Tallies: " << N_RINGS << " rings x " << N_KBINS << " energy bins" << std::endl;
    std::cerr << "Events: " << g_n_events << std::endl << std::endl;

    runManager->BeamOn(g_n_events);

    // --- Summary ---
    std::cerr << std::endl;
    std::cerr << "Photons escaping target (k > " << K_MIN << " MeV): "
              << g_total_photons << std::endl;
    std::cerr << "Photons per electron: "
              << (double)g_total_photons / g_n_events << std::endl;
    std::cerr << "Energy deposited: " << g_dose_mev << " MeV total, "
              << g_dose_mev / g_n_events << " MeV/electron" << std::endl;
    std::cerr << "Backscattered e-: " << g_backscatter_count << " ("
              << 100.0 * g_backscatter_count / g_n_events << "%)" << std::endl;

    // --- CSV output ---
    std::cout << "ring,theta_lo,theta_hi,theta_mid,solid_angle,"
              << "k_lo,k_hi,k_center,dk,"
              << "n_photons,intensity,rel_error" << std::endl;

    for (int r = 0; r < N_RINGS; r++) {
        double th_lo = r * D_THETA;
        double th_hi = (r + 1) * D_THETA;
        double th_mid = (th_lo + th_hi) / 2.0;
        double cos_lo = std::cos(th_lo * M_PI / 180.0);
        double cos_hi = std::cos(th_hi * M_PI / 180.0);
        double solid_angle = 2.0 * M_PI * std::abs(cos_lo - cos_hi);

        for (int k = 0; k < N_KBINS; k++) {
            double k_lo = g_kbin_edges[k];
            double k_hi = g_kbin_edges[k + 1];
            double k_center = std::sqrt(k_lo * k_hi);
            double dk = k_hi - k_lo;

            long n = g_counts[r][k];
            double intensity = 0;
            double rel_err = 1.0;
            if (n > 0) {
                // I(k, theta) = k * N_photons / (N_electrons * dk * solid_angle)
                // Units: MeV / (MeV sr electron)
                intensity = k_center * (double)n / ((double)g_n_events * dk * solid_angle);
                rel_err = 1.0 / std::sqrt((double)n);
            }

            std::cout << r << "," << th_lo << "," << th_hi << "," << th_mid << ","
                      << solid_angle << ","
                      << k_lo << "," << k_hi << "," << k_center << "," << dk << ","
                      << n << "," << intensity << "," << rel_err << std::endl;
        }
    }

    delete runManager;
    return 0;
}
