#!/bin/bash
#SBATCH --job-name=julia_warmup
#SBATCH --partition=cip-ws
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=warmup.out
#SBATCH --error=warmup.err

set -euo pipefail
module load julia/1.11.3
cd $SLURM_SUBMIT_DIR

export JULIA_PROJECT=./env_nethive_multiverse

SCRATCH_BASE=/scratch/n/N.Pfaffenzeller
DEPOT_COMPILED="$SCRATCH_BASE/.julia_depot_compiled_shared"
DEPOT_SHARED="$SCRATCH_BASE/.julia_depot_shared"
mkdir -p "$DEPOT_COMPILED" "$DEPOT_SHARED"
export JULIA_DEPOT_PATH="$DEPOT_COMPILED:$DEPOT_SHARED"
echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"

# optional, but reduces surprises:
export JULIA_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0

export DATADEPS_ALWAYS_ACCEPT=1
export DATADEPS_LOAD_PATH="$SCRATCH_BASE/datadeps"
mkdir -p "$DATADEPS_LOAD_PATH"

julia --project=./env_nethive_multiverse -e '
using Pkg
Pkg.instantiate()
Pkg.precompile()

# sanity: load the heavy stack once
using Flux, MLDatasets, DataFrames, CSV, JSON3, Distributions, ArgParse, JSON3

# trigger the CIFAR10 download into DATADEPS_LOAD_PATH (non-interactive)
CIFAR10(:train)

println("Warmup done.")
'
