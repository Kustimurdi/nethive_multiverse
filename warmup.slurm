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
export JULIA_DEPOT_PATH=/scratch/n/N.Pfaffenzeller/.julia_depot_shared
mkdir -p "$JULIA_DEPOT_PATH"

# optional, but reduces surprises:
export JULIA_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0

export DATADEPS_ALWAYS_ACCEPT=1
export DATADEPS_LOAD_PATH=/scratch/n/N.Pfaffenzeller/datadeps
mkdir -p "$DATADEPS_LOAD_PATH"


julia --project=./env_nethive_multiverse -e '
using Pkg
Pkg.instantiate()
Pkg.precompile()
# sanity: load the heavy stack once
using Flux, MLDatasets, DataFrames, CSV, JSON3, Distributions
CIFAR10(:train)
println("Warmup done.")
'
