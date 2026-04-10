/// Geant4 thick-target bremsstrahlung: Fe 3 MeV with kill sphere
///
/// Geometry:
///   - Fe cylinder: 3.6 mm thick, 9.6 mm radius (CSDA range + 1mm margin)
///   - Vacuum kill sphere: 200 mm radius, 1 mm thick shell (sensitive detector)
///   - Beam: 1 cm diameter disk, 3 MeV electrons along +z, starts at z=-5mm
///
/// Scoring at kill sphere:
///   - Every PHOTON: energy, theta, phi (momentum direction)
///   - Every ELECTRON: energy, theta, phi, position (x,y,z)
///   - Separate binary files for photons and electrons
///
/// Scoring in target (stepping action):
///   - Total energy deposited (dose)
///   - Electron backscatter count (electrons leaving front face)
///
/// Binary output format:
///   Header: 3 ints (n_events, record_size_bytes, version)
///   Photon records: k_MeV, theta_deg, phi_deg  (3 x float32 = 12 bytes)
///   Electron records: k_MeV, theta_deg, phi_deg, x_mm, y_mm, z_mm (6 x float32 = 24 bytes)
///
/// Usage: ./g4env.sh ./fe_3mev_killsphere [n_events] [prefix]
///        Default: 10000000 events, prefix="fe_3mev"
///        Outputs: {prefix}_photons.bin, {prefix}_electrons.bin, {prefix}_summary.txt

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
#include "G4VisAttributes.hh"
#include "G4Colour.hh"

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

// =====================================================================
// Configuration
// =====================================================================
static const double TARGET_Z  = 3.6;    // mm
static const double TARGET_R  = 9.6;    // mm
static const double BEAM_R    = 5.0;    // mm
static const double T0_MEV    = 3.0;
static const double SPHERE_R  = 200.0;  // mm
static const double SPHERE_DR = 1.0;    // mm

static int g_n_events = 10000000;
static std::string g_prefix = "fe_3mev";

// =====================================================================
// Record structs
// =====================================================================
#pragma pack(push, 1)
struct PhotonRec {
    float k_MeV;       // photon energy
    float theta_deg;   // polar angle from +z (beam axis)
    float phi_deg;     // azimuthal angle
};  // 12 bytes

struct ElectronRec {
    float k_MeV;       // kinetic energy
    float theta_deg;   // polar angle from +z
    float phi_deg;     // azimuthal angle
    float x_mm;        // position on kill sphere
    float y_mm;
    float z_mm;
};  // 24 bytes
#pragma pack(pop)

// =====================================================================
// Global tallies
// =====================================================================
static std::mutex g_photon_mutex;
static std::mutex g_electron_mutex;
static std::mutex g_dose_mutex;
static std::ofstream g_photon_file;
static std::ofstream g_electron_file;
static std::atomic<long> g_photon_count{0};
static std::atomic<long> g_electron_count{0};
static std::atomic<long> g_backscatter_count{0};
static double g_dose_mev = 0.0;        // total energy deposited in target
static double g_dose_sq_mev = 0.0;     // sum of squares for variance

// =====================================================================
// Kill Sphere Sensitive Detector
// =====================================================================
class KillSphereSD : public G4VSensitiveDetector {
public:
    KillSphereSD(const G4String& name) : G4VSensitiveDetector(name) {}

    G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override {
        auto* track = step->GetTrack();
        double k = track->GetKineticEnergy() / MeV;

        // Momentum direction for angular binning
        G4ThreeVector dir = track->GetMomentumDirection();
        double cos_th = std::max(-1.0, std::min(1.0, dir.z()));
        double theta = std::acos(cos_th) * 180.0 / M_PI;
        double phi = std::atan2(dir.y(), dir.x()) * 180.0 / M_PI;
        if (phi < 0) phi += 360.0;

        if (track->GetDefinition() == G4Gamma::Definition() && k > 0.001) {
            PhotonRec rec;
            rec.k_MeV = (float)k;
            rec.theta_deg = (float)theta;
            rec.phi_deg = (float)phi;
            {
                std::lock_guard<std::mutex> lock(g_photon_mutex);
                g_photon_file.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
            }
            g_photon_count++;
        }
        else if (track->GetDefinition() == G4Electron::Definition() && k > 0.001) {
            G4ThreeVector pos = step->GetPreStepPoint()->GetPosition();
            ElectronRec rec;
            rec.k_MeV = (float)k;
            rec.theta_deg = (float)theta;
            rec.phi_deg = (float)phi;
            rec.x_mm = (float)(pos.x() / mm);
            rec.y_mm = (float)(pos.y() / mm);
            rec.z_mm = (float)(pos.z() / mm);
            {
                std::lock_guard<std::mutex> lock(g_electron_mutex);
                g_electron_file.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
            }
            g_electron_count++;
        }

        // Kill everything at the sphere
        track->SetTrackStatus(fStopAndKill);
        return true;
    }

    void Initialize(G4HCofThisEvent*) override {}
    void EndOfEvent(G4HCofThisEvent*) override {}
};

// =====================================================================
// Stepping action: dose in target + backscatter
// =====================================================================
class SteppingAction : public G4UserSteppingAction {
public:
    void UserSteppingAction(const G4Step* step) override {
        auto* preVol = step->GetPreStepPoint()->GetPhysicalVolume();
        if (!preVol || preVol->GetName() != "Target") return;

        double edep = step->GetTotalEnergyDeposit() / MeV;
        if (edep > 0) {
            std::lock_guard<std::mutex> lock(g_dose_mutex);
            g_dose_mev += edep;
            g_dose_sq_mev += edep * edep;
        }

        // Electron backscatter: e- leaving target through front face (z < 0.01 mm)
        auto* track = step->GetTrack();
        auto* postVol = step->GetPostStepPoint()->GetPhysicalVolume();
        if (track->GetDefinition() == G4Electron::Definition() &&
            postVol && postVol->GetName() == "World") {
            if (step->GetPostStepPoint()->GetPosition().z() < 0.01 * mm) {
                g_backscatter_count++;
            }
        }
    }
};

// =====================================================================
// Detector Construction
// =====================================================================
class DetectorConstruction : public G4VUserDetectorConstruction {
public:
    G4VPhysicalVolume* Construct() override {
        auto* nist = G4NistManager::Instance();
        auto* fe = nist->FindOrBuildMaterial("G4_Fe");
        auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");

        double world_half = (SPHERE_R + 50) * mm;
        auto* world_solid = new G4Box("World", world_half, world_half, world_half);
        auto* world_log = new G4LogicalVolume(world_solid, vacuum, "World");
        auto* world_phys = new G4PVPlacement(nullptr, {}, world_log,
                                              "World", nullptr, false, 0);

        // Fe target: front face at z=0
        auto* target_solid = new G4Tubs("Target", 0, TARGET_R * mm,
                                         TARGET_Z / 2 * mm, 0, 360 * deg);
        auto* target_log = new G4LogicalVolume(target_solid, fe, "Target");
        new G4PVPlacement(nullptr,
                          G4ThreeVector(0, 0, TARGET_Z / 2 * mm),
                          target_log, "Target", world_log, false, 0);

        // Kill sphere centered on target midpoint
        auto* sphere_solid = new G4Sphere("KillSphere",
                                           SPHERE_R * mm,
                                           (SPHERE_R + SPHERE_DR) * mm,
                                           0, 360 * deg, 0, 180 * deg);
        auto* sphere_log = new G4LogicalVolume(sphere_solid, vacuum, "KillSphere");
        new G4PVPlacement(nullptr,
                          G4ThreeVector(0, 0, TARGET_Z / 2 * mm),
                          sphere_log, "KillSphere", world_log, false, 0);

        return world_phys;
    }

    void ConstructSDandField() override {
        auto* sd = new KillSphereSD("KillSphereSD");
        G4SDManager::GetSDMpointer()->AddNewDetector(sd);
        SetSensitiveDetector("KillSphere", sd);
    }
};

// =====================================================================
// Beam
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
        double r = BEAM_R * mm * std::sqrt(G4UniformRand());
        double phi = 2.0 * M_PI * G4UniformRand();
        fGun->SetParticlePosition(
            G4ThreeVector(r * std::cos(phi), r * std::sin(phi), -5.0 * mm));
        fGun->GeneratePrimaryVertex(evt);
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
    if (argc > 2) g_prefix = argv[2];

    std::string photon_path = g_prefix + "_photons.bin";
    std::string electron_path = g_prefix + "_electrons.bin";
    std::string summary_path = g_prefix + "_summary.txt";

    // Open binary output files
    g_photon_file.open(photon_path, std::ios::binary);
    g_electron_file.open(electron_path, std::ios::binary);
    if (!g_photon_file || !g_electron_file) {
        std::cerr << "Cannot open output files" << std::endl;
        return 1;
    }

    // Write headers
    int phot_header[3] = { g_n_events, (int)sizeof(PhotonRec), 2 };  // version 2
    g_photon_file.write(reinterpret_cast<const char*>(phot_header), sizeof(phot_header));

    int elec_header[3] = { g_n_events, (int)sizeof(ElectronRec), 2 };
    g_electron_file.write(reinterpret_cast<const char*>(elec_header), sizeof(elec_header));

    // Build and run
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

    std::cerr << "=== Fe 3 MeV Kill Sphere — Full Tally ===" << std::endl;
    std::cerr << "Target: Fe " << TARGET_Z << " mm x " << TARGET_R << " mm R" << std::endl;
    std::cerr << "Beam: " << 2*BEAM_R << " mm dia, " << T0_MEV << " MeV e-" << std::endl;
    std::cerr << "Kill sphere: " << SPHERE_R << " mm radius" << std::endl;
    std::cerr << "Events: " << g_n_events << std::endl;
    std::cerr << "Output: " << photon_path << ", " << electron_path << std::endl;
    std::cerr << std::endl;

    runManager->BeamOn(g_n_events);

    g_photon_file.close();
    g_electron_file.close();

    // Compute statistics
    long np = g_photon_count.load();
    long ne = g_electron_count.load();
    long nb = g_backscatter_count.load();
    double dose_mean = g_dose_mev / g_n_events;
    double dose_var = g_dose_sq_mev / g_n_events - dose_mean * dose_mean;
    double dose_std = std::sqrt(std::max(dose_var, 0.0));

    // Summary to stderr
    std::cerr << std::endl;
    std::cerr << "=== Results ===" << std::endl;
    std::cerr << "Photons at kill sphere:   " << np
              << "  (" << (double)np/g_n_events << " per electron)" << std::endl;
    std::cerr << "Electrons at kill sphere: " << ne
              << "  (" << (double)ne/g_n_events << " per electron)" << std::endl;
    std::cerr << "Backscattered electrons:  " << nb
              << "  (" << 100.0*nb/g_n_events << "%)" << std::endl;
    std::cerr << "Dose in target: " << dose_mean << " +/- " << dose_std
              << " MeV/electron" << std::endl;
    std::cerr << "Energy balance: beam=" << T0_MEV << " MeV, deposited=" << dose_mean
              << ", escaping=" << T0_MEV - dose_mean << " MeV/electron" << std::endl;
    std::cerr << "Photon file:   " << photon_path << " ("
              << (12 + np*sizeof(PhotonRec))/1024 << " KB)" << std::endl;
    std::cerr << "Electron file: " << electron_path << " ("
              << (12 + ne*sizeof(ElectronRec))/1024 << " KB)" << std::endl;

    // Summary to text file
    std::ofstream sf(summary_path);
    sf << "Fe 3 MeV Kill Sphere Simulation Summary" << std::endl;
    sf << "========================================" << std::endl;
    sf << "Target: Fe, " << TARGET_Z << " mm thick, " << TARGET_R << " mm radius" << std::endl;
    sf << "Beam: " << 2*BEAM_R << " mm dia, " << T0_MEV << " MeV electrons" << std::endl;
    sf << "Kill sphere: " << SPHERE_R << " mm radius" << std::endl;
    sf << "Events: " << g_n_events << std::endl;
    sf << "Physics: G4EmStandardPhysics_option4 (Seltzer-Berger)" << std::endl;
    sf << std::endl;
    sf << "Photons at sphere:   " << np << "  (" << (double)np/g_n_events << "/e-)" << std::endl;
    sf << "Electrons at sphere: " << ne << "  (" << (double)ne/g_n_events << "/e-)" << std::endl;
    sf << "Backscatter (front): " << nb << "  (" << 100.0*nb/g_n_events << "%)" << std::endl;
    sf << "Dose: " << dose_mean << " +/- " << dose_std << " MeV/electron" << std::endl;
    sf << "Energy balance: " << T0_MEV << " = " << dose_mean << " (deposited) + "
       << T0_MEV - dose_mean << " (escaping)" << std::endl;
    sf.close();

    delete runManager;
    return 0;
}
