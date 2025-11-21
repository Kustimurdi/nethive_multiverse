#!/bin/bash
#SBATCH --job-name=bee_sim_array
#SBATCH --partition=th-ws
#SBATCH --time=05:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-36%36
#SBATCH --output=2t2b_high_deadtime/cifar10svhn210000e_sweep/logs/job_%A_%a.out
#SBATCH --error=2t2b_high_deadtime/cifar10svhn210000e_sweep/logs/job_%A_%a.err

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
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_1.json"
    BASE_NAME="param_1_rep_1"
    ;;
  2)
    echo "Running parameter combination 2, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_2.json"
    BASE_NAME="param_2_rep_1"
    ;;
  3)
    echo "Running parameter combination 3, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_3.json"
    BASE_NAME="param_3_rep_1"
    ;;
  4)
    echo "Running parameter combination 4, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_4.json"
    BASE_NAME="param_4_rep_1"
    ;;
  5)
    echo "Running parameter combination 5, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_5.json"
    BASE_NAME="param_5_rep_1"
    ;;
  6)
    echo "Running parameter combination 6, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_6.json"
    BASE_NAME="param_6_rep_1"
    ;;
  7)
    echo "Running parameter combination 7, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_7.json"
    BASE_NAME="param_7_rep_1"
    ;;
  8)
    echo "Running parameter combination 8, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_8.json"
    BASE_NAME="param_8_rep_1"
    ;;
  9)
    echo "Running parameter combination 9, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_9.json"
    BASE_NAME="param_9_rep_1"
    ;;
  10)
    echo "Running parameter combination 10, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_10.json"
    BASE_NAME="param_10_rep_1"
    ;;
  11)
    echo "Running parameter combination 11, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_11.json"
    BASE_NAME="param_11_rep_1"
    ;;
  12)
    echo "Running parameter combination 12, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_12.json"
    BASE_NAME="param_12_rep_1"
    ;;
  13)
    echo "Running parameter combination 13, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_13.json"
    BASE_NAME="param_13_rep_1"
    ;;
  14)
    echo "Running parameter combination 14, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_14.json"
    BASE_NAME="param_14_rep_1"
    ;;
  15)
    echo "Running parameter combination 15, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_15.json"
    BASE_NAME="param_15_rep_1"
    ;;
  16)
    echo "Running parameter combination 16, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_16.json"
    BASE_NAME="param_16_rep_1"
    ;;
  17)
    echo "Running parameter combination 17, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_17.json"
    BASE_NAME="param_17_rep_1"
    ;;
  18)
    echo "Running parameter combination 18, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_18.json"
    BASE_NAME="param_18_rep_1"
    ;;
  19)
    echo "Running parameter combination 19, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_19.json"
    BASE_NAME="param_19_rep_1"
    ;;
  20)
    echo "Running parameter combination 20, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_20.json"
    BASE_NAME="param_20_rep_1"
    ;;
  21)
    echo "Running parameter combination 21, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_21.json"
    BASE_NAME="param_21_rep_1"
    ;;
  22)
    echo "Running parameter combination 22, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_22.json"
    BASE_NAME="param_22_rep_1"
    ;;
  23)
    echo "Running parameter combination 23, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_23.json"
    BASE_NAME="param_23_rep_1"
    ;;
  24)
    echo "Running parameter combination 24, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_24.json"
    BASE_NAME="param_24_rep_1"
    ;;
  25)
    echo "Running parameter combination 25, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_25.json"
    BASE_NAME="param_25_rep_1"
    ;;
  26)
    echo "Running parameter combination 26, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_26.json"
    BASE_NAME="param_26_rep_1"
    ;;
  27)
    echo "Running parameter combination 27, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_27.json"
    BASE_NAME="param_27_rep_1"
    ;;
  28)
    echo "Running parameter combination 28, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_28.json"
    BASE_NAME="param_28_rep_1"
    ;;
  29)
    echo "Running parameter combination 29, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_29.json"
    BASE_NAME="param_29_rep_1"
    ;;
  30)
    echo "Running parameter combination 30, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_30.json"
    BASE_NAME="param_30_rep_1"
    ;;
  31)
    echo "Running parameter combination 31, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_31.json"
    BASE_NAME="param_31_rep_1"
    ;;
  32)
    echo "Running parameter combination 32, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_32.json"
    BASE_NAME="param_32_rep_1"
    ;;
  33)
    echo "Running parameter combination 33, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_33.json"
    BASE_NAME="param_33_rep_1"
    ;;
  34)
    echo "Running parameter combination 34, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_34.json"
    BASE_NAME="param_34_rep_1"
    ;;
  35)
    echo "Running parameter combination 35, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_35.json"
    BASE_NAME="param_35_rep_1"
    ;;
  36)
    echo "Running parameter combination 36, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10svhn210000e_sweep/configs/config_36.json"
    BASE_NAME="param_36_rep_1"
    ;;
  *)
    echo "Error: Unknown array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

# Run simulation
echo "Starting Julia simulation..."
julia run_simulation.jl --config $CONFIG_FILE --output-dir 2t2b_high_deadtime/cifar10svhn210000e_sweep/data --base-name $BASE_NAME --timestamp --verbose --save-results

echo "Job finished at: $(date)"
