/// Geant4 visualization: Fe 3 MeV beam hitting iron slab
/// Shows the kill sphere geometry used in the production simulation.
/// Zoomed to target scale so you can see electron tracks and photon emission.
///
/// Usage: ./g4env.sh ./fe_3mev_vis

#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
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

#include <cmath>
#include <string>

static const double TARGET_Z  = 3.6;    // mm
static const double TARGET_R  = 9.6;    // mm
static const double BEAM_R    = 5.0;    // mm
static const double T0_MEV    = 3.0;
static const double SPHERE_R  = 200.0;  // mm
static const double SPHERE_DR = 1.0;    // mm

// Kill sphere SD — kills photons on contact (same as production)
class KillSphereSD : public G4VSensitiveDetector {
public:
    KillSphereSD(const G4String& name) : G4VSensitiveDetector(name) {}
    G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override {
        step->GetTrack()->SetTrackStatus(fStopAndKill);
        return true;
    }
    void Initialize(G4HCofThisEvent*) override {}
    void EndOfEvent(G4HCofThisEvent*) override {}
};

class DetectorConstruction : public G4VUserDetectorConstruction {
public:
    G4VPhysicalVolume* Construct() override {
        auto* nist = G4NistManager::Instance();
        auto* fe = nist->FindOrBuildMaterial("G4_Fe");
        auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");

        // World
        double world_half = (SPHERE_R + 50) * mm;
        auto* world_solid = new G4Box("World", world_half, world_half, world_half);
        auto* world_log = new G4LogicalVolume(world_solid, vacuum, "World");
        world_log->SetVisAttributes(G4VisAttributes::GetInvisible());
        auto* world_phys = new G4PVPlacement(nullptr, {}, world_log,
                                              "World", nullptr, false, 0);

        // Fe target: front face at z=0, back face at z=3.6 mm
        auto* target_solid = new G4Tubs("Target", 0, TARGET_R * mm,
                                         TARGET_Z / 2 * mm, 0, 360 * deg);
        auto* target_log = new G4LogicalVolume(target_solid, fe, "Target");
        auto* tvis = new G4VisAttributes(G4Colour(0.4, 0.4, 0.7, 0.8));
        tvis->SetForceSolid(true);
        target_log->SetVisAttributes(tvis);
        new G4PVPlacement(nullptr,
                          G4ThreeVector(0, 0, TARGET_Z / 2 * mm),
                          target_log, "Target", world_log, false, 0);

        // Kill sphere: thin shell at 200 mm
        auto* sphere_solid = new G4Sphere("KillSphere",
                                           SPHERE_R * mm,
                                           (SPHERE_R + SPHERE_DR) * mm,
                                           0, 360 * deg, 0, 180 * deg);
        auto* sphere_log = new G4LogicalVolume(sphere_solid, vacuum, "KillSphere");
        auto* svis = new G4VisAttributes(G4Colour(0.2, 0.8, 0.2, 0.08));
        svis->SetForceWireframe(true);
        sphere_log->SetVisAttributes(svis);
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

// 1 cm diameter beam, uniform disk, 3 MeV electrons along +z
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

class PhysicsList : public G4VModularPhysicsList {
public:
    PhysicsList() {
        RegisterPhysics(new G4EmStandardPhysics_option4());
        RegisterPhysics(new G4DecayPhysics());
    }
};

int main(int argc, char** argv) {
    auto* ui = new G4UIExecutive(argc, argv);

    auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
    runManager->SetUserInitialization(new DetectorConstruction());
    runManager->SetUserInitialization(new PhysicsList());
    runManager->SetUserAction(new PrimaryGenerator());
    runManager->Initialize();

    auto* visManager = new G4VisExecutive();
    visManager->Initialize();

    auto* uiMgr = G4UImanager::GetUIpointer();

    // Open viewer
    uiMgr->ApplyCommand("/vis/open OGL 1200x800");
    uiMgr->ApplyCommand("/vis/viewer/set/autoRefresh false");
    uiMgr->ApplyCommand("/vis/verbose errors");

    // Draw geometry
    uiMgr->ApplyCommand("/vis/drawVolume");

    // Zoom to target scale — target is 3.6 mm, sphere is 200 mm
    // Start zoomed in so the iron slab fills the view
    uiMgr->ApplyCommand("/vis/viewer/set/viewpointThetaPhi 60 30 deg");
    uiMgr->ApplyCommand("/vis/viewer/zoomTo 15");
    uiMgr->ApplyCommand("/vis/viewer/set/targetPoint 0 0 1.8 mm");

    // Style
    uiMgr->ApplyCommand("/vis/viewer/set/lightsVector -1 0.5 0.5");
    uiMgr->ApplyCommand("/vis/viewer/set/background 0.05 0.05 0.1 1");

    // Trajectories — color by particle
    uiMgr->ApplyCommand("/vis/scene/add/trajectories smooth");
    uiMgr->ApplyCommand("/vis/modeling/trajectories/create/drawByParticleID");
    uiMgr->ApplyCommand("/vis/modeling/trajectories/drawByParticleID-0/set e- red");
    uiMgr->ApplyCommand("/vis/modeling/trajectories/drawByParticleID-0/set e+ blue");
    uiMgr->ApplyCommand("/vis/modeling/trajectories/drawByParticleID-0/set gamma green");

    // Accumulate tracks from multiple events
    uiMgr->ApplyCommand("/vis/scene/endOfEventAction accumulate 200");

    uiMgr->ApplyCommand("/vis/viewer/set/autoRefresh true");
    uiMgr->ApplyCommand("/vis/viewer/flush");

    // Run 200 events — enough to see the beam pattern and photon spray
    uiMgr->ApplyCommand("/run/beamOn 200");

    ui->SessionStart();

    delete visManager;
    delete runManager;
    delete ui;
    return 0;
}
