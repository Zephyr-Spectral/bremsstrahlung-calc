#!/bin/bash
# Build the generalized kill sphere executable for Geant4 batch runs.
# Usage: ./build_killsphere.sh <source.cc> <output_exe>
#
# Uses geant4-config from the geant4_env conda environment directly
# without conda activate (avoids Geant4 activation script bugs).

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <source.cc> <output_exe>" >&2
    exit 1
fi

SRC="$1"
EXE="$2"

if [ ! -f "$SRC" ]; then
    echo "Source file not found: $SRC" >&2
    exit 1
fi

# --- Locate geant4-config directly in the conda env ---
G4ENV_BIN=""
for candidate in \
    "/opt/homebrew/Caskroom/miniforge/base/envs/geant4_env/bin" \
    "$HOME/miniforge3/envs/geant4_env/bin" \
    "$HOME/miniconda3/envs/geant4_env/bin"; do
    if [ -x "${candidate}/geant4-config" ]; then
        G4ENV_BIN="$candidate"
        break
    fi
done

if [ -z "$G4ENV_BIN" ]; then
    echo "ERROR: geant4-config not found in any geant4_env location." >&2
    exit 1
fi

G4CONFIG="${G4ENV_BIN}/geant4-config"
G4PREFIX=$("$G4CONFIG" --prefix)
G4VERSION=$("$G4CONFIG" --version)

# Extract only -D and -I flags
CFLAGS=$("$G4CONFIG" --cflags | tr ' ' '\n' | grep -E '^-[DI]' | tr '\n' ' ')
LIBS=$("$G4CONFIG" --libs)

echo "Compiling $(basename "$SRC") -> $(basename "$EXE")"
echo "  Geant4 ${G4VERSION} at ${G4PREFIX}"

c++ -std=c++17 -O2 \
    $CFLAGS \
    -I"${G4PREFIX}/include/Geant4" \
    -I"${G4PREFIX}/include" \
    "$SRC" \
    $LIBS \
    -Wl,-rpath,"${G4PREFIX}/lib" \
    -o "$EXE"

echo "Built: $(basename "$EXE")"
