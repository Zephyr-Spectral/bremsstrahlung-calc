/// Geant4 thick-target bremsstrahlung validation
/// Scores energy + angle of photons escaping the target rear face.
/// Usage: ./thick_target_brems <material> <T0_MeV> <n_events>
/// Output: CSV to stdout  (k_MeV, theta_deg, weight)

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4UserRunAction.hh"
#include "G4UserEventAction.hh"
#include "G4UserSteppingAction.hh"

#include "G4Box.hh"
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

#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <mutex>
#include <vector>

// --- Global config (set from main) ---
static std::string g_material = "G4_Pb";
static double g_T0_MeV = 3.0;
static double g_thickness_cm = 0.0;  // set from CSDA range

// --- Output storage ---
static std::mutex g_mutex;
struct PhotonRecord { double k_MeV; double theta_deg; };
static std::vector<PhotonRecord> g_photons;

// =====================================================================
// Detector: slab of material, thick enough to stop electrons
// =====================================================================
class DetectorConstruction : public G4VUserDetectorConstruction {
public:
    G4VPhysicalVolume* Construct() override {
        auto* nist = G4NistManager::Instance();
        auto* mat = nist->FindOrBuildMaterial(g_material);

        // Get CSDA range to set thickness
        // Use a generous 1.5x the nominal range
        double range_cm = 0.5;  // fallback
        // We'll set this from config
        if (g_thickness_cm > 0) range_cm = g_thickness_cm;

        double slab_z = range_cm * cm;
        double slab_xy = 10.0 * cm;  // wide enough laterally

        // World
        auto* world_solid = new G4Box("World", slab_xy, slab_xy, 2*slab_z);
        auto* world_log = new G4LogicalVolume(world_solid,
            nist->FindOrBuildMaterial("G4_Galactic"), "World");
        auto* world_phys = new G4PVPlacement(nullptr, {}, world_log, "World", nullptr, false, 0);

        // Target slab centered at z=slab_z/2 (front face at z=0)
        auto* slab_solid = new G4Box("Slab", slab_xy, slab_xy, slab_z/2);
        auto* slab_log = new G4LogicalVolume(slab_solid, mat, "Slab");
        new G4PVPlacement(nullptr, G4ThreeVector(0, 0, slab_z/2), slab_log, "Slab", world_log, false, 0);

        return world_phys;
    }
};

// =====================================================================
// Primary generator: pencil electron beam along +z
// =====================================================================
class PrimaryGenerator : public G4VUserPrimaryGeneratorAction {
    G4ParticleGun* fGun;
public:
    PrimaryGenerator() : fGun(new G4ParticleGun(1)) {
        fGun->SetParticleDefinition(G4Electron::Definition());
        fGun->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
        fGun->SetParticlePosition(G4ThreeVector(0, 0, -0.01*mm));
        fGun->SetParticleEnergy(g_T0_MeV * MeV);
    }
    ~PrimaryGenerator() override { delete fGun; }
    void GeneratePrimaries(G4Event* evt) override { fGun->GeneratePrimaryVertex(evt); }
};

// =====================================================================
// Stepping action: record photons leaving the target rear face
// =====================================================================
class SteppingAction : public G4UserSteppingAction {
public:
    void UserSteppingAction(const G4Step* step) override {
        auto* track = step->GetTrack();

        // Only photons
        if (track->GetDefinition() != G4Gamma::Definition()) return;

        // Check if photon is leaving the slab volume into world
        auto* preVol = step->GetPreStepPoint()->GetPhysicalVolume();
        auto* postVol = step->GetPostStepPoint()->GetPhysicalVolume();
        if (!preVol || !postVol) return;
        if (preVol->GetName() != "Slab" || postVol->GetName() != "World") return;

        // Photon escaping from target
        double k = track->GetKineticEnergy() / MeV;
        G4ThreeVector dir = track->GetMomentumDirection();
        double theta = std::acos(dir.z()) * 180.0 / M_PI;  // angle from beam axis

        // Only forward hemisphere (transmitted through rear)
        auto postPos = step->GetPostStepPoint()->GetPosition();
        if (postPos.z() <= 0) return;  // skip backward-escaping photons

        std::lock_guard<std::mutex> lock(g_mutex);
        g_photons.push_back({k, theta});
    }
};

// =====================================================================
// Physics list: option4 (best low-energy EM accuracy)
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
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <G4_material> <T0_MeV> <n_events> [thickness_cm]" << std::endl;
        std::cerr << "Example: " << argv[0] << " G4_Pb 3.0 1000000 0.45" << std::endl;
        return 1;
    }

    g_material = argv[1];
    g_T0_MeV = std::stod(argv[2]);
    int n_events = std::stoi(argv[3]);
    if (argc > 4) g_thickness_cm = std::stod(argv[4]);

    // Suppress Geant4 output
    G4UImanager::GetUIpointer()->ApplyCommand("/run/verbose 0");
    G4UImanager::GetUIpointer()->ApplyCommand("/event/verbose 0");
    G4UImanager::GetUIpointer()->ApplyCommand("/tracking/verbose 0");

    auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
    runManager->SetUserInitialization(new DetectorConstruction());
    runManager->SetUserInitialization(new PhysicsList());
    runManager->SetUserAction(new PrimaryGenerator());
    runManager->SetUserAction(new SteppingAction());
    runManager->Initialize();

    // Suppress output during run
    G4UImanager::GetUIpointer()->ApplyCommand("/run/verbose 0");
    G4UImanager::GetUIpointer()->ApplyCommand("/run/printProgress 100000");

    runManager->BeamOn(n_events);

    // Output CSV
    std::cerr << "# Photons recorded: " << g_photons.size() << std::endl;
    std::cout << "k_MeV,theta_deg" << std::endl;
    for (auto& p : g_photons) {
        std::cout << p.k_MeV << "," << p.theta_deg << std::endl;
    }

    delete runManager;
    return 0;
}
