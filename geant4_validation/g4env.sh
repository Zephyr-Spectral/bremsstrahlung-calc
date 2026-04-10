#!/bin/bash
# Geant4 environment wrapper — sets data paths and library path, then exec's the command.
# Usage: ./g4env.sh ./thick_target_brems G4_Pb 3.0 100000 0.45

G4PREFIX=/opt/homebrew/Caskroom/miniforge/base/envs/geant4_env
G4D=$G4PREFIX/share/Geant4/data

export G4ENSDFSTATEDATA=$G4D/ENSDFSTATE3.0
export G4LEDATA=$G4D/EMLOW8.6.1
export G4LEVELGAMMADATA=$G4D/PhotonEvaporation6.1
export G4RADIOACTIVEDATA=$G4D/RadioactiveDecay6.1.2
export G4PARTICLEXSDATA=$G4D/PARTICLEXS4.1
export G4NEUTRONHPDATA=$G4D/NDL4.7.1
export G4SAIDXSDATA=$G4D/SAIDDATA2.0
export G4ABLADATA=$G4D/ABLA3.3
export G4INCLDATA=$G4D/INCL1.2
export G4REALSURFACEDATA=$G4D/RealSurface2.2
export DYLD_LIBRARY_PATH=$G4PREFIX/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}

exec "$@"
