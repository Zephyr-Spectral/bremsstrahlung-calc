#!/bin/bash
# Build the generalized kill sphere executable for Geant4 batch runs.
# Usage: ./build_killsphere.sh <source.cc> <output_exe>
#
# Requires: conda geant4_env with Geant4 11.3+ installed.
# Called by batch_run.py — not intended for direct use.

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

# Activate Geant4 conda environment
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate geant4_env 2>/dev/null

G4PREFIX=$(geant4-config --prefix)

# Extract only -D and -I flags from geant4-config --cflags
CFLAGS=$(geant4-config --cflags | tr ' ' '\n' | grep -E '^-[DI]' | tr '\n' ' ')
LIBS=$(geant4-config --libs)

echo "Compiling $SRC -> $EXE"
echo "  G4 prefix: $G4PREFIX"
echo "  G4 version: $(geant4-config --version)"

c++ -std=c++17 -O2 \
    $CFLAGS \
    -I"${G4PREFIX}/include/Geant4" \
    -I"${G4PREFIX}/include" \
    "$SRC" \
    $LIBS \
    -Wl,-rpath,"${G4PREFIX}/lib" \
    -o "$EXE"

echo "Built: $EXE"
