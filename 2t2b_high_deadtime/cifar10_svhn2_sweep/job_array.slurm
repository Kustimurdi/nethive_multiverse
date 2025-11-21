#!/bin/bash
#SBATCH --job-name=bee_sim_array
#SBATCH --partition=th-ws
#SBATCH --time=05:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-72%72
#SBATCH --output=2t2b_high_deadtime/cifar10_svhn2_sweep/logs/job_%A_%a.out
#SBATCH --error=2t2b_high_deadtime/cifar10_svhn2_sweep/logs/job_%A_%a.err

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
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_1.json"
    BASE_NAME="param_1_rep_1"
    ;;
  2)
    echo "Running parameter combination 2, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_2.json"
    BASE_NAME="param_2_rep_1"
    ;;
  3)
    echo "Running parameter combination 3, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_3.json"
    BASE_NAME="param_3_rep_1"
    ;;
  4)
    echo "Running parameter combination 4, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_4.json"
    BASE_NAME="param_4_rep_1"
    ;;
  5)
    echo "Running parameter combination 5, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_5.json"
    BASE_NAME="param_5_rep_1"
    ;;
  6)
    echo "Running parameter combination 6, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_6.json"
    BASE_NAME="param_6_rep_1"
    ;;
  7)
    echo "Running parameter combination 7, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_7.json"
    BASE_NAME="param_7_rep_1"
    ;;
  8)
    echo "Running parameter combination 8, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_8.json"
    BASE_NAME="param_8_rep_1"
    ;;
  9)
    echo "Running parameter combination 9, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_9.json"
    BASE_NAME="param_9_rep_1"
    ;;
  10)
    echo "Running parameter combination 10, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_10.json"
    BASE_NAME="param_10_rep_1"
    ;;
  11)
    echo "Running parameter combination 11, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_11.json"
    BASE_NAME="param_11_rep_1"
    ;;
  12)
    echo "Running parameter combination 12, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_12.json"
    BASE_NAME="param_12_rep_1"
    ;;
  13)
    echo "Running parameter combination 13, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_13.json"
    BASE_NAME="param_13_rep_1"
    ;;
  14)
    echo "Running parameter combination 14, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_14.json"
    BASE_NAME="param_14_rep_1"
    ;;
  15)
    echo "Running parameter combination 15, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_15.json"
    BASE_NAME="param_15_rep_1"
    ;;
  16)
    echo "Running parameter combination 16, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_16.json"
    BASE_NAME="param_16_rep_1"
    ;;
  17)
    echo "Running parameter combination 17, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_17.json"
    BASE_NAME="param_17_rep_1"
    ;;
  18)
    echo "Running parameter combination 18, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_18.json"
    BASE_NAME="param_18_rep_1"
    ;;
  19)
    echo "Running parameter combination 19, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_19.json"
    BASE_NAME="param_19_rep_1"
    ;;
  20)
    echo "Running parameter combination 20, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_20.json"
    BASE_NAME="param_20_rep_1"
    ;;
  21)
    echo "Running parameter combination 21, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_21.json"
    BASE_NAME="param_21_rep_1"
    ;;
  22)
    echo "Running parameter combination 22, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_22.json"
    BASE_NAME="param_22_rep_1"
    ;;
  23)
    echo "Running parameter combination 23, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_23.json"
    BASE_NAME="param_23_rep_1"
    ;;
  24)
    echo "Running parameter combination 24, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_24.json"
    BASE_NAME="param_24_rep_1"
    ;;
  25)
    echo "Running parameter combination 25, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_25.json"
    BASE_NAME="param_25_rep_1"
    ;;
  26)
    echo "Running parameter combination 26, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_26.json"
    BASE_NAME="param_26_rep_1"
    ;;
  27)
    echo "Running parameter combination 27, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_27.json"
    BASE_NAME="param_27_rep_1"
    ;;
  28)
    echo "Running parameter combination 28, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_28.json"
    BASE_NAME="param_28_rep_1"
    ;;
  29)
    echo "Running parameter combination 29, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_29.json"
    BASE_NAME="param_29_rep_1"
    ;;
  30)
    echo "Running parameter combination 30, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_30.json"
    BASE_NAME="param_30_rep_1"
    ;;
  31)
    echo "Running parameter combination 31, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_31.json"
    BASE_NAME="param_31_rep_1"
    ;;
  32)
    echo "Running parameter combination 32, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_32.json"
    BASE_NAME="param_32_rep_1"
    ;;
  33)
    echo "Running parameter combination 33, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_33.json"
    BASE_NAME="param_33_rep_1"
    ;;
  34)
    echo "Running parameter combination 34, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_34.json"
    BASE_NAME="param_34_rep_1"
    ;;
  35)
    echo "Running parameter combination 35, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_35.json"
    BASE_NAME="param_35_rep_1"
    ;;
  36)
    echo "Running parameter combination 36, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_36.json"
    BASE_NAME="param_36_rep_1"
    ;;
  37)
    echo "Running parameter combination 37, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_37.json"
    BASE_NAME="param_37_rep_1"
    ;;
  38)
    echo "Running parameter combination 38, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_38.json"
    BASE_NAME="param_38_rep_1"
    ;;
  39)
    echo "Running parameter combination 39, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_39.json"
    BASE_NAME="param_39_rep_1"
    ;;
  40)
    echo "Running parameter combination 40, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_40.json"
    BASE_NAME="param_40_rep_1"
    ;;
  41)
    echo "Running parameter combination 41, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_41.json"
    BASE_NAME="param_41_rep_1"
    ;;
  42)
    echo "Running parameter combination 42, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_42.json"
    BASE_NAME="param_42_rep_1"
    ;;
  43)
    echo "Running parameter combination 43, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_43.json"
    BASE_NAME="param_43_rep_1"
    ;;
  44)
    echo "Running parameter combination 44, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_44.json"
    BASE_NAME="param_44_rep_1"
    ;;
  45)
    echo "Running parameter combination 45, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_45.json"
    BASE_NAME="param_45_rep_1"
    ;;
  46)
    echo "Running parameter combination 46, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_46.json"
    BASE_NAME="param_46_rep_1"
    ;;
  47)
    echo "Running parameter combination 47, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_47.json"
    BASE_NAME="param_47_rep_1"
    ;;
  48)
    echo "Running parameter combination 48, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_48.json"
    BASE_NAME="param_48_rep_1"
    ;;
  49)
    echo "Running parameter combination 49, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_49.json"
    BASE_NAME="param_49_rep_1"
    ;;
  50)
    echo "Running parameter combination 50, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_50.json"
    BASE_NAME="param_50_rep_1"
    ;;
  51)
    echo "Running parameter combination 51, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_51.json"
    BASE_NAME="param_51_rep_1"
    ;;
  52)
    echo "Running parameter combination 52, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_52.json"
    BASE_NAME="param_52_rep_1"
    ;;
  53)
    echo "Running parameter combination 53, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_53.json"
    BASE_NAME="param_53_rep_1"
    ;;
  54)
    echo "Running parameter combination 54, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_54.json"
    BASE_NAME="param_54_rep_1"
    ;;
  55)
    echo "Running parameter combination 55, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_55.json"
    BASE_NAME="param_55_rep_1"
    ;;
  56)
    echo "Running parameter combination 56, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_56.json"
    BASE_NAME="param_56_rep_1"
    ;;
  57)
    echo "Running parameter combination 57, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_57.json"
    BASE_NAME="param_57_rep_1"
    ;;
  58)
    echo "Running parameter combination 58, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_58.json"
    BASE_NAME="param_58_rep_1"
    ;;
  59)
    echo "Running parameter combination 59, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_59.json"
    BASE_NAME="param_59_rep_1"
    ;;
  60)
    echo "Running parameter combination 60, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_60.json"
    BASE_NAME="param_60_rep_1"
    ;;
  61)
    echo "Running parameter combination 61, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_61.json"
    BASE_NAME="param_61_rep_1"
    ;;
  62)
    echo "Running parameter combination 62, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_62.json"
    BASE_NAME="param_62_rep_1"
    ;;
  63)
    echo "Running parameter combination 63, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_63.json"
    BASE_NAME="param_63_rep_1"
    ;;
  64)
    echo "Running parameter combination 64, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_64.json"
    BASE_NAME="param_64_rep_1"
    ;;
  65)
    echo "Running parameter combination 65, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_65.json"
    BASE_NAME="param_65_rep_1"
    ;;
  66)
    echo "Running parameter combination 66, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_66.json"
    BASE_NAME="param_66_rep_1"
    ;;
  67)
    echo "Running parameter combination 67, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_67.json"
    BASE_NAME="param_67_rep_1"
    ;;
  68)
    echo "Running parameter combination 68, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_68.json"
    BASE_NAME="param_68_rep_1"
    ;;
  69)
    echo "Running parameter combination 69, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_69.json"
    BASE_NAME="param_69_rep_1"
    ;;
  70)
    echo "Running parameter combination 70, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_70.json"
    BASE_NAME="param_70_rep_1"
    ;;
  71)
    echo "Running parameter combination 71, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_71.json"
    BASE_NAME="param_71_rep_1"
    ;;
  72)
    echo "Running parameter combination 72, replicate 1"
    CONFIG_FILE="2t2b_high_deadtime/cifar10_svhn2_sweep/configs/config_72.json"
    BASE_NAME="param_72_rep_1"
    ;;
  *)
    echo "Error: Unknown array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

# Run simulation
echo "Starting Julia simulation..."
julia run_simulation.jl --config $CONFIG_FILE --output-dir 2t2b_high_deadtime/cifar10_svhn2_sweep/data --base-name $BASE_NAME --timestamp --verbose --save-results

echo "Job finished at: $(date)"
