#!/bin/bash
#SBATCH --job-name=bee_sim_array
#SBATCH --partition=th-ws
#SBATCH --time=05:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-6%6
#SBATCH --output=single_tasks/svhn2_500e_sweep/logs/job_%A_%a.out
#SBATCH --error=single_tasks/svhn2_500e_sweep/logs/job_%A_%a.err

# Job array information
echo "Job array started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"

# Load modules (required for cluster compatibility)
echo "Loading Julia module..."
module load julia/1.11.3
echo "Module loaded: $(which julia)"

# Change to project directory
echo "Changing to project directory..."
cd $SLURM_SUBMIT_DIR
echo "Current directory: $(pwd)"

# Set Julia project environment
export JULIA_PROJECT=./env_nethive_multiverse

# Configuration mapping based on array task ID
case $SLURM_ARRAY_TASK_ID in
  1)
    echo "Running parameter combination 1, replicate 1"
    CONFIG_FILE="single_tasks/svhn2_500e_sweep/configs/config_1.json"
    BASE_NAME="param_1_rep_1"
    ;;
  2)
    echo "Running parameter combination 2, replicate 1"
    CONFIG_FILE="single_tasks/svhn2_500e_sweep/configs/config_2.json"
    BASE_NAME="param_2_rep_1"
    ;;
  3)
    echo "Running parameter combination 3, replicate 1"
    CONFIG_FILE="single_tasks/svhn2_500e_sweep/configs/config_3.json"
    BASE_NAME="param_3_rep_1"
    ;;
  4)
    echo "Running parameter combination 4, replicate 1"
    CONFIG_FILE="single_tasks/svhn2_500e_sweep/configs/config_4.json"
    BASE_NAME="param_4_rep_1"
    ;;
  5)
    echo "Running parameter combination 5, replicate 1"
    CONFIG_FILE="single_tasks/svhn2_500e_sweep/configs/config_5.json"
    BASE_NAME="param_5_rep_1"
    ;;
  6)
    echo "Running parameter combination 6, replicate 1"
    CONFIG_FILE="single_tasks/svhn2_500e_sweep/configs/config_6.json"
    BASE_NAME="param_6_rep_1"
    ;;
  *)
    echo "Error: Unknown array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

# Run simulation
echo "Starting Julia simulation..."
julia run_simulation.jl --config $CONFIG_FILE --output-dir single_tasks/svhn2_500e_sweep/data --base-name $BASE_NAME --timestamp --verbose --save-results

echo "Job finished at: $(date)"
