/// Geant4 thick-target bremsstrahlung: Fe 3 MeV with ring detectors
///
/// Geometry:
///   - Fe cylinder: 3.6 mm thick (CSDA range + 1mm), 9.6 mm radius
///   - Beam: 1 cm diameter pencil, 3 MeV electrons along +z
///   - 36 ring detectors on a 200 mm radius sphere, 5 deg each (0-180)
///   - Each ring is a G4Sphere segment capturing all azimuthal photons
///
/// Scoring per ring:
///   - Photon energy spectrum (40 log bins, 10 keV to 3 MeV)
///   - Total photon count, energy fluence
///   - Energy deposited in target (dose check)
///
/// Output: CSV with columns:
///   ring_id, theta_lo, theta_hi, k_bin_lo, k_bin_hi, k_center,
///   n_photons, intensity, stat_error
///
/// Usage: ./fe_3mev_rings [n_events]  (default 10000000)

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4UserEventAction.hh"
#include "G4UserRunAction.hh"

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
#include "G4StepPoint.hh"
#include "G4RunManager.hh"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <mutex>
#include <numeric>

// =====================================================================
// Constants
// =====================================================================
static const int    N_RINGS   = 36;       // 5 deg per ring, 0-180
static const double D_THETA   = 5.0;      // degrees per ring
static const int    N_KBINS   = 40;       // energy bins
static const double K_MIN     = 0.010;    // MeV
static const double K_MAX     = 3.000;    // MeV
static const double DET_R     = 200.0;    // mm - detector sphere radius

// Fe target
static const double TARGET_Z  = 3.6;      // mm thickness (CSDA range + 1mm)
static const double TARGET_R  = 9.6;      // mm radius
static const double BEAM_R    = 5.0;      // mm beam radius (1 cm diameter)
static const double T0_MEV    = 3.0;

static int g_n_events = 10000000;

// =====================================================================
// Tally storage — thread-safe
// =====================================================================
static std::mutex g_mutex;

// photon_counts[ring][kbin] — number of photons
static std::vector<std::vector<long>> g_counts(N_RINGS, std::vector<long>(N_KBINS, 0));
// sum of weights squared for variance: sum_w2[ring][kbin]
static std::vector<std::vector<double>> g_sum_w2(N_RINGS, std::vector<double>(N_KBINS, 0.0));
// total energy deposited in target (MeV) for dose check
static double g_dose_mev = 0.0;
// electron backscatter count
static long g_backscatter_count = 0;

// Log-spaced energy bin edges
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
    // Binary search in log-spaced bins
    double log_k = std::log10(k_mev);
    double log_min = std::log10(K_MIN);
    double log_max = std::log10(K_MAX);
    int bin = (int)((log_k - log_min) / (log_max - log_min) * N_KBINS);
    if (bin < 0) bin = 0;
    if (bin >= N_KBINS) bin = N_KBINS - 1;
    return bin;
}

int find_ring(double theta_deg) {
    if (theta_deg < 0 || theta_deg >= 180.0) return -1;
    return (int)(theta_deg / D_THETA);
}

// =====================================================================
// Detector Construction
// =====================================================================
class DetectorConstruction : public G4VUserDetectorConstruction {
public:
    G4VPhysicalVolume* Construct() override {
        auto* nist = G4NistManager::Instance();
        auto* fe = nist->FindOrBuildMaterial("G4_Fe");
        auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");

        // World: large enough to contain detector sphere
        double world_half = DET_R * 1.5 * mm;
        auto* world_solid = new G4Box("World", world_half, world_half, world_half);
        auto* world_log = new G4LogicalVolume(world_solid, vacuum, "World");
        auto* world_phys = new G4PVPlacement(nullptr, {}, world_log, "World",
                                              nullptr, false, 0);

        // Fe target cylinder: front face at z=0, extends to z=TARGET_Z
        auto* target_solid = new G4Tubs("Target", 0, TARGET_R*mm,
                                         TARGET_Z/2*mm, 0, 360*deg);
        auto* target_log = new G4LogicalVolume(target_solid, fe, "Target");
        new G4PVPlacement(nullptr,
                          G4ThreeVector(0, 0, TARGET_Z/2*mm),  // center at z=TARGET_Z/2
                          target_log, "Target", world_log, false, 0);

        // Ring detectors: thin spherical shell segments
        // Centered on the target midpoint (z = TARGET_Z/2)
        // Each ring: G4Sphere from theta_lo to theta_hi
        double shell_inner = DET_R * mm;
        double shell_outer = (DET_R + 1.0) * mm;  // 1mm thick shell

        for (int i = 0; i < N_RINGS; i++) {
            double theta_lo = i * D_THETA * deg;
            double theta_hi = (i + 1) * D_THETA * deg;
            std::string name = "Ring_" + std::to_string(i);

            auto* ring_solid = new G4Sphere(name, shell_inner, shell_outer,
                                             0, 360*deg, theta_lo, D_THETA*deg);
            auto* ring_log = new G4LogicalVolume(ring_solid, vacuum, name);
            // Place centered on target midpoint
            new G4PVPlacement(nullptr,
                              G4ThreeVector(0, 0, TARGET_Z/2*mm),
                              ring_log, name, world_log, false, i);
        }

        return world_phys;
    }
};

// =====================================================================
// Primary Generator: 1 cm diameter beam, 3 MeV electrons along +z
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
        // Random position within 1 cm diameter circle
        double r = BEAM_R * mm * std::sqrt(G4UniformRand());
        double phi = 2.0 * M_PI * G4UniformRand();
        double x = r * std::cos(phi);
        double y = r * std::sin(phi);
        fGun->SetParticlePosition(G4ThreeVector(x, y, -0.1*mm));
        fGun->GeneratePrimaryVertex(evt);
    }
};

// =====================================================================
// Stepping Action: score photons entering ring volumes + dose in target
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

        // Score energy deposited in target (for dose check)
        if (preName == "Target") {
            double edep = step->GetTotalEnergyDeposit() / MeV;
            if (edep > 0) {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_dose_mev += edep;
            }
        }

        // Score electron backscatter (electron leaving target back to world at z<0)
        if (track->GetDefinition() == G4Electron::Definition() &&
            preName == "Target" && postName == "World") {
            auto pos = step->GetPostStepPoint()->GetPosition();
            if (pos.z() < 0) {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_backscatter_count++;
            }
        }

        // Score photons entering any Ring_ volume
        if (track->GetDefinition() != G4Gamma::Definition()) return;
        if (postName.substr(0, 5) != "Ring_") return;
        if (preName.substr(0, 5) == "Ring_") return;  // already in a ring

        double k_mev = track->GetKineticEnergy() / MeV;
        int kbin = find_kbin(k_mev);
        if (kbin < 0) return;

        int ring_id = postVol->GetCopyNo();
        if (ring_id < 0 || ring_id >= N_RINGS) return;

        std::lock_guard<std::mutex> lock(g_mutex);
        g_counts[ring_id][kbin]++;
        g_sum_w2[ring_id][kbin] += 1.0;  // weight=1 for analog MC

        // Kill the photon after scoring to avoid double-counting
        // (it would exit the thin shell and re-enter next ring)
        track->SetTrackStatus(fStopAndKill);
    }
};

// =====================================================================
// Physics: option4 for best EM accuracy
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

    std::cerr << "=== Fe 3 MeV Ring Detector Simulation ===" << std::endl;
    std::cerr << "Target: Fe cylinder " << TARGET_Z << " mm thick, "
              << TARGET_R << " mm radius" << std::endl;
    std::cerr << "Beam: " << 2*BEAM_R << " mm diameter, " << T0_MEV << " MeV electrons" << std::endl;
    std::cerr << "Detectors: " << N_RINGS << " rings x " << N_KBINS
              << " energy bins on " << DET_R << " mm sphere" << std::endl;
    std::cerr << "Events: " << g_n_events << std::endl;
    std::cerr << std::endl;

    runManager->BeamOn(g_n_events);

    // --- Output results ---

    // Summary to stderr
    long total_photons = 0;
    for (int r = 0; r < N_RINGS; r++)
        for (int k = 0; k < N_KBINS; k++)
            total_photons += g_counts[r][k];

    std::cerr << std::endl;
    std::cerr << "Total scored photons: " << total_photons << std::endl;
    std::cerr << "Energy deposited in target: " << g_dose_mev << " MeV" << std::endl;
    std::cerr << "Backscattered electrons: " << g_backscatter_count
              << " (" << 100.0*g_backscatter_count/g_n_events << "%)" << std::endl;
    std::cerr << "Mean dose per electron: " << g_dose_mev/g_n_events << " MeV" << std::endl;

    // CSV to stdout
    std::cout << "ring_id,theta_lo,theta_hi,k_lo,k_hi,k_center,"
              << "n_photons,intensity,rel_error" << std::endl;

    for (int r = 0; r < N_RINGS; r++) {
        double th_lo = r * D_THETA;
        double th_hi = (r + 1) * D_THETA;
        double cos_lo = std::cos(th_lo * M_PI / 180.0);
        double cos_hi = std::cos(th_hi * M_PI / 180.0);
        double solid_angle = 2.0 * M_PI * std::abs(cos_lo - cos_hi);

        for (int k = 0; k < N_KBINS; k++) {
            double k_lo = g_kbin_edges[k];
            double k_hi = g_kbin_edges[k + 1];
            double k_center = std::sqrt(k_lo * k_hi);  // geometric mean
            double dk = k_hi - k_lo;

            long n = g_counts[r][k];
            // Intensity: I = k * n / (N_elec * dk * solid_angle)
            // Units: MeV / (MeV sr electron)
            double intensity = 0;
            double rel_err = 0;
            if (n > 0) {
                intensity = k_center * n / ((double)g_n_events * dk * solid_angle);
                rel_err = 1.0 / std::sqrt((double)n);  // Poisson statistics
            }

            std::cout << r << "," << th_lo << "," << th_hi << ","
                      << k_lo << "," << k_hi << "," << k_center << ","
                      << n << "," << intensity << "," << rel_err << std::endl;
        }
    }

    delete runManager;
    return 0;
}
