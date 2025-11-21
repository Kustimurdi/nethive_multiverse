#!/bin/bash
#SBATCH --job-name=bee_sim_array
#SBATCH --partition=th-ws
#SBATCH --time=10:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-192%100
#SBATCH --output=5t5b/5t5b_mfcsb/logs/job_%A_%a.out
#SBATCH --error=5t5b/5t5b_mfcsb/logs/job_%A_%a.err

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
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_1.json"
    BASE_NAME="param_1_rep_1"
    ;;
  2)
    echo "Running parameter combination 2, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_2.json"
    BASE_NAME="param_2_rep_1"
    ;;
  3)
    echo "Running parameter combination 3, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_3.json"
    BASE_NAME="param_3_rep_1"
    ;;
  4)
    echo "Running parameter combination 4, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_4.json"
    BASE_NAME="param_4_rep_1"
    ;;
  5)
    echo "Running parameter combination 5, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_5.json"
    BASE_NAME="param_5_rep_1"
    ;;
  6)
    echo "Running parameter combination 6, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_6.json"
    BASE_NAME="param_6_rep_1"
    ;;
  7)
    echo "Running parameter combination 7, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_7.json"
    BASE_NAME="param_7_rep_1"
    ;;
  8)
    echo "Running parameter combination 8, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_8.json"
    BASE_NAME="param_8_rep_1"
    ;;
  9)
    echo "Running parameter combination 9, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_9.json"
    BASE_NAME="param_9_rep_1"
    ;;
  10)
    echo "Running parameter combination 10, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_10.json"
    BASE_NAME="param_10_rep_1"
    ;;
  11)
    echo "Running parameter combination 11, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_11.json"
    BASE_NAME="param_11_rep_1"
    ;;
  12)
    echo "Running parameter combination 12, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_12.json"
    BASE_NAME="param_12_rep_1"
    ;;
  13)
    echo "Running parameter combination 13, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_13.json"
    BASE_NAME="param_13_rep_1"
    ;;
  14)
    echo "Running parameter combination 14, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_14.json"
    BASE_NAME="param_14_rep_1"
    ;;
  15)
    echo "Running parameter combination 15, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_15.json"
    BASE_NAME="param_15_rep_1"
    ;;
  16)
    echo "Running parameter combination 16, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_16.json"
    BASE_NAME="param_16_rep_1"
    ;;
  17)
    echo "Running parameter combination 17, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_17.json"
    BASE_NAME="param_17_rep_1"
    ;;
  18)
    echo "Running parameter combination 18, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_18.json"
    BASE_NAME="param_18_rep_1"
    ;;
  19)
    echo "Running parameter combination 19, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_19.json"
    BASE_NAME="param_19_rep_1"
    ;;
  20)
    echo "Running parameter combination 20, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_20.json"
    BASE_NAME="param_20_rep_1"
    ;;
  21)
    echo "Running parameter combination 21, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_21.json"
    BASE_NAME="param_21_rep_1"
    ;;
  22)
    echo "Running parameter combination 22, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_22.json"
    BASE_NAME="param_22_rep_1"
    ;;
  23)
    echo "Running parameter combination 23, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_23.json"
    BASE_NAME="param_23_rep_1"
    ;;
  24)
    echo "Running parameter combination 24, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_24.json"
    BASE_NAME="param_24_rep_1"
    ;;
  25)
    echo "Running parameter combination 25, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_25.json"
    BASE_NAME="param_25_rep_1"
    ;;
  26)
    echo "Running parameter combination 26, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_26.json"
    BASE_NAME="param_26_rep_1"
    ;;
  27)
    echo "Running parameter combination 27, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_27.json"
    BASE_NAME="param_27_rep_1"
    ;;
  28)
    echo "Running parameter combination 28, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_28.json"
    BASE_NAME="param_28_rep_1"
    ;;
  29)
    echo "Running parameter combination 29, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_29.json"
    BASE_NAME="param_29_rep_1"
    ;;
  30)
    echo "Running parameter combination 30, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_30.json"
    BASE_NAME="param_30_rep_1"
    ;;
  31)
    echo "Running parameter combination 31, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_31.json"
    BASE_NAME="param_31_rep_1"
    ;;
  32)
    echo "Running parameter combination 32, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_32.json"
    BASE_NAME="param_32_rep_1"
    ;;
  33)
    echo "Running parameter combination 33, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_33.json"
    BASE_NAME="param_33_rep_1"
    ;;
  34)
    echo "Running parameter combination 34, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_34.json"
    BASE_NAME="param_34_rep_1"
    ;;
  35)
    echo "Running parameter combination 35, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_35.json"
    BASE_NAME="param_35_rep_1"
    ;;
  36)
    echo "Running parameter combination 36, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_36.json"
    BASE_NAME="param_36_rep_1"
    ;;
  37)
    echo "Running parameter combination 37, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_37.json"
    BASE_NAME="param_37_rep_1"
    ;;
  38)
    echo "Running parameter combination 38, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_38.json"
    BASE_NAME="param_38_rep_1"
    ;;
  39)
    echo "Running parameter combination 39, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_39.json"
    BASE_NAME="param_39_rep_1"
    ;;
  40)
    echo "Running parameter combination 40, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_40.json"
    BASE_NAME="param_40_rep_1"
    ;;
  41)
    echo "Running parameter combination 41, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_41.json"
    BASE_NAME="param_41_rep_1"
    ;;
  42)
    echo "Running parameter combination 42, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_42.json"
    BASE_NAME="param_42_rep_1"
    ;;
  43)
    echo "Running parameter combination 43, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_43.json"
    BASE_NAME="param_43_rep_1"
    ;;
  44)
    echo "Running parameter combination 44, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_44.json"
    BASE_NAME="param_44_rep_1"
    ;;
  45)
    echo "Running parameter combination 45, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_45.json"
    BASE_NAME="param_45_rep_1"
    ;;
  46)
    echo "Running parameter combination 46, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_46.json"
    BASE_NAME="param_46_rep_1"
    ;;
  47)
    echo "Running parameter combination 47, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_47.json"
    BASE_NAME="param_47_rep_1"
    ;;
  48)
    echo "Running parameter combination 48, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_48.json"
    BASE_NAME="param_48_rep_1"
    ;;
  49)
    echo "Running parameter combination 49, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_49.json"
    BASE_NAME="param_49_rep_1"
    ;;
  50)
    echo "Running parameter combination 50, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_50.json"
    BASE_NAME="param_50_rep_1"
    ;;
  51)
    echo "Running parameter combination 51, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_51.json"
    BASE_NAME="param_51_rep_1"
    ;;
  52)
    echo "Running parameter combination 52, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_52.json"
    BASE_NAME="param_52_rep_1"
    ;;
  53)
    echo "Running parameter combination 53, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_53.json"
    BASE_NAME="param_53_rep_1"
    ;;
  54)
    echo "Running parameter combination 54, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_54.json"
    BASE_NAME="param_54_rep_1"
    ;;
  55)
    echo "Running parameter combination 55, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_55.json"
    BASE_NAME="param_55_rep_1"
    ;;
  56)
    echo "Running parameter combination 56, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_56.json"
    BASE_NAME="param_56_rep_1"
    ;;
  57)
    echo "Running parameter combination 57, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_57.json"
    BASE_NAME="param_57_rep_1"
    ;;
  58)
    echo "Running parameter combination 58, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_58.json"
    BASE_NAME="param_58_rep_1"
    ;;
  59)
    echo "Running parameter combination 59, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_59.json"
    BASE_NAME="param_59_rep_1"
    ;;
  60)
    echo "Running parameter combination 60, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_60.json"
    BASE_NAME="param_60_rep_1"
    ;;
  61)
    echo "Running parameter combination 61, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_61.json"
    BASE_NAME="param_61_rep_1"
    ;;
  62)
    echo "Running parameter combination 62, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_62.json"
    BASE_NAME="param_62_rep_1"
    ;;
  63)
    echo "Running parameter combination 63, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_63.json"
    BASE_NAME="param_63_rep_1"
    ;;
  64)
    echo "Running parameter combination 64, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_64.json"
    BASE_NAME="param_64_rep_1"
    ;;
  65)
    echo "Running parameter combination 65, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_65.json"
    BASE_NAME="param_65_rep_1"
    ;;
  66)
    echo "Running parameter combination 66, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_66.json"
    BASE_NAME="param_66_rep_1"
    ;;
  67)
    echo "Running parameter combination 67, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_67.json"
    BASE_NAME="param_67_rep_1"
    ;;
  68)
    echo "Running parameter combination 68, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_68.json"
    BASE_NAME="param_68_rep_1"
    ;;
  69)
    echo "Running parameter combination 69, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_69.json"
    BASE_NAME="param_69_rep_1"
    ;;
  70)
    echo "Running parameter combination 70, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_70.json"
    BASE_NAME="param_70_rep_1"
    ;;
  71)
    echo "Running parameter combination 71, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_71.json"
    BASE_NAME="param_71_rep_1"
    ;;
  72)
    echo "Running parameter combination 72, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_72.json"
    BASE_NAME="param_72_rep_1"
    ;;
  73)
    echo "Running parameter combination 73, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_73.json"
    BASE_NAME="param_73_rep_1"
    ;;
  74)
    echo "Running parameter combination 74, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_74.json"
    BASE_NAME="param_74_rep_1"
    ;;
  75)
    echo "Running parameter combination 75, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_75.json"
    BASE_NAME="param_75_rep_1"
    ;;
  76)
    echo "Running parameter combination 76, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_76.json"
    BASE_NAME="param_76_rep_1"
    ;;
  77)
    echo "Running parameter combination 77, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_77.json"
    BASE_NAME="param_77_rep_1"
    ;;
  78)
    echo "Running parameter combination 78, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_78.json"
    BASE_NAME="param_78_rep_1"
    ;;
  79)
    echo "Running parameter combination 79, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_79.json"
    BASE_NAME="param_79_rep_1"
    ;;
  80)
    echo "Running parameter combination 80, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_80.json"
    BASE_NAME="param_80_rep_1"
    ;;
  81)
    echo "Running parameter combination 81, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_81.json"
    BASE_NAME="param_81_rep_1"
    ;;
  82)
    echo "Running parameter combination 82, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_82.json"
    BASE_NAME="param_82_rep_1"
    ;;
  83)
    echo "Running parameter combination 83, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_83.json"
    BASE_NAME="param_83_rep_1"
    ;;
  84)
    echo "Running parameter combination 84, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_84.json"
    BASE_NAME="param_84_rep_1"
    ;;
  85)
    echo "Running parameter combination 85, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_85.json"
    BASE_NAME="param_85_rep_1"
    ;;
  86)
    echo "Running parameter combination 86, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_86.json"
    BASE_NAME="param_86_rep_1"
    ;;
  87)
    echo "Running parameter combination 87, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_87.json"
    BASE_NAME="param_87_rep_1"
    ;;
  88)
    echo "Running parameter combination 88, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_88.json"
    BASE_NAME="param_88_rep_1"
    ;;
  89)
    echo "Running parameter combination 89, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_89.json"
    BASE_NAME="param_89_rep_1"
    ;;
  90)
    echo "Running parameter combination 90, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_90.json"
    BASE_NAME="param_90_rep_1"
    ;;
  91)
    echo "Running parameter combination 91, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_91.json"
    BASE_NAME="param_91_rep_1"
    ;;
  92)
    echo "Running parameter combination 92, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_92.json"
    BASE_NAME="param_92_rep_1"
    ;;
  93)
    echo "Running parameter combination 93, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_93.json"
    BASE_NAME="param_93_rep_1"
    ;;
  94)
    echo "Running parameter combination 94, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_94.json"
    BASE_NAME="param_94_rep_1"
    ;;
  95)
    echo "Running parameter combination 95, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_95.json"
    BASE_NAME="param_95_rep_1"
    ;;
  96)
    echo "Running parameter combination 96, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_96.json"
    BASE_NAME="param_96_rep_1"
    ;;
  97)
    echo "Running parameter combination 97, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_97.json"
    BASE_NAME="param_97_rep_1"
    ;;
  98)
    echo "Running parameter combination 98, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_98.json"
    BASE_NAME="param_98_rep_1"
    ;;
  99)
    echo "Running parameter combination 99, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_99.json"
    BASE_NAME="param_99_rep_1"
    ;;
  100)
    echo "Running parameter combination 100, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_100.json"
    BASE_NAME="param_100_rep_1"
    ;;
  101)
    echo "Running parameter combination 101, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_101.json"
    BASE_NAME="param_101_rep_1"
    ;;
  102)
    echo "Running parameter combination 102, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_102.json"
    BASE_NAME="param_102_rep_1"
    ;;
  103)
    echo "Running parameter combination 103, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_103.json"
    BASE_NAME="param_103_rep_1"
    ;;
  104)
    echo "Running parameter combination 104, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_104.json"
    BASE_NAME="param_104_rep_1"
    ;;
  105)
    echo "Running parameter combination 105, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_105.json"
    BASE_NAME="param_105_rep_1"
    ;;
  106)
    echo "Running parameter combination 106, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_106.json"
    BASE_NAME="param_106_rep_1"
    ;;
  107)
    echo "Running parameter combination 107, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_107.json"
    BASE_NAME="param_107_rep_1"
    ;;
  108)
    echo "Running parameter combination 108, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_108.json"
    BASE_NAME="param_108_rep_1"
    ;;
  109)
    echo "Running parameter combination 109, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_109.json"
    BASE_NAME="param_109_rep_1"
    ;;
  110)
    echo "Running parameter combination 110, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_110.json"
    BASE_NAME="param_110_rep_1"
    ;;
  111)
    echo "Running parameter combination 111, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_111.json"
    BASE_NAME="param_111_rep_1"
    ;;
  112)
    echo "Running parameter combination 112, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_112.json"
    BASE_NAME="param_112_rep_1"
    ;;
  113)
    echo "Running parameter combination 113, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_113.json"
    BASE_NAME="param_113_rep_1"
    ;;
  114)
    echo "Running parameter combination 114, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_114.json"
    BASE_NAME="param_114_rep_1"
    ;;
  115)
    echo "Running parameter combination 115, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_115.json"
    BASE_NAME="param_115_rep_1"
    ;;
  116)
    echo "Running parameter combination 116, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_116.json"
    BASE_NAME="param_116_rep_1"
    ;;
  117)
    echo "Running parameter combination 117, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_117.json"
    BASE_NAME="param_117_rep_1"
    ;;
  118)
    echo "Running parameter combination 118, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_118.json"
    BASE_NAME="param_118_rep_1"
    ;;
  119)
    echo "Running parameter combination 119, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_119.json"
    BASE_NAME="param_119_rep_1"
    ;;
  120)
    echo "Running parameter combination 120, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_120.json"
    BASE_NAME="param_120_rep_1"
    ;;
  121)
    echo "Running parameter combination 121, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_121.json"
    BASE_NAME="param_121_rep_1"
    ;;
  122)
    echo "Running parameter combination 122, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_122.json"
    BASE_NAME="param_122_rep_1"
    ;;
  123)
    echo "Running parameter combination 123, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_123.json"
    BASE_NAME="param_123_rep_1"
    ;;
  124)
    echo "Running parameter combination 124, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_124.json"
    BASE_NAME="param_124_rep_1"
    ;;
  125)
    echo "Running parameter combination 125, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_125.json"
    BASE_NAME="param_125_rep_1"
    ;;
  126)
    echo "Running parameter combination 126, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_126.json"
    BASE_NAME="param_126_rep_1"
    ;;
  127)
    echo "Running parameter combination 127, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_127.json"
    BASE_NAME="param_127_rep_1"
    ;;
  128)
    echo "Running parameter combination 128, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_128.json"
    BASE_NAME="param_128_rep_1"
    ;;
  129)
    echo "Running parameter combination 129, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_129.json"
    BASE_NAME="param_129_rep_1"
    ;;
  130)
    echo "Running parameter combination 130, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_130.json"
    BASE_NAME="param_130_rep_1"
    ;;
  131)
    echo "Running parameter combination 131, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_131.json"
    BASE_NAME="param_131_rep_1"
    ;;
  132)
    echo "Running parameter combination 132, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_132.json"
    BASE_NAME="param_132_rep_1"
    ;;
  133)
    echo "Running parameter combination 133, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_133.json"
    BASE_NAME="param_133_rep_1"
    ;;
  134)
    echo "Running parameter combination 134, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_134.json"
    BASE_NAME="param_134_rep_1"
    ;;
  135)
    echo "Running parameter combination 135, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_135.json"
    BASE_NAME="param_135_rep_1"
    ;;
  136)
    echo "Running parameter combination 136, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_136.json"
    BASE_NAME="param_136_rep_1"
    ;;
  137)
    echo "Running parameter combination 137, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_137.json"
    BASE_NAME="param_137_rep_1"
    ;;
  138)
    echo "Running parameter combination 138, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_138.json"
    BASE_NAME="param_138_rep_1"
    ;;
  139)
    echo "Running parameter combination 139, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_139.json"
    BASE_NAME="param_139_rep_1"
    ;;
  140)
    echo "Running parameter combination 140, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_140.json"
    BASE_NAME="param_140_rep_1"
    ;;
  141)
    echo "Running parameter combination 141, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_141.json"
    BASE_NAME="param_141_rep_1"
    ;;
  142)
    echo "Running parameter combination 142, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_142.json"
    BASE_NAME="param_142_rep_1"
    ;;
  143)
    echo "Running parameter combination 143, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_143.json"
    BASE_NAME="param_143_rep_1"
    ;;
  144)
    echo "Running parameter combination 144, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_144.json"
    BASE_NAME="param_144_rep_1"
    ;;
  145)
    echo "Running parameter combination 145, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_145.json"
    BASE_NAME="param_145_rep_1"
    ;;
  146)
    echo "Running parameter combination 146, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_146.json"
    BASE_NAME="param_146_rep_1"
    ;;
  147)
    echo "Running parameter combination 147, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_147.json"
    BASE_NAME="param_147_rep_1"
    ;;
  148)
    echo "Running parameter combination 148, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_148.json"
    BASE_NAME="param_148_rep_1"
    ;;
  149)
    echo "Running parameter combination 149, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_149.json"
    BASE_NAME="param_149_rep_1"
    ;;
  150)
    echo "Running parameter combination 150, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_150.json"
    BASE_NAME="param_150_rep_1"
    ;;
  151)
    echo "Running parameter combination 151, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_151.json"
    BASE_NAME="param_151_rep_1"
    ;;
  152)
    echo "Running parameter combination 152, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_152.json"
    BASE_NAME="param_152_rep_1"
    ;;
  153)
    echo "Running parameter combination 153, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_153.json"
    BASE_NAME="param_153_rep_1"
    ;;
  154)
    echo "Running parameter combination 154, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_154.json"
    BASE_NAME="param_154_rep_1"
    ;;
  155)
    echo "Running parameter combination 155, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_155.json"
    BASE_NAME="param_155_rep_1"
    ;;
  156)
    echo "Running parameter combination 156, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_156.json"
    BASE_NAME="param_156_rep_1"
    ;;
  157)
    echo "Running parameter combination 157, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_157.json"
    BASE_NAME="param_157_rep_1"
    ;;
  158)
    echo "Running parameter combination 158, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_158.json"
    BASE_NAME="param_158_rep_1"
    ;;
  159)
    echo "Running parameter combination 159, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_159.json"
    BASE_NAME="param_159_rep_1"
    ;;
  160)
    echo "Running parameter combination 160, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_160.json"
    BASE_NAME="param_160_rep_1"
    ;;
  161)
    echo "Running parameter combination 161, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_161.json"
    BASE_NAME="param_161_rep_1"
    ;;
  162)
    echo "Running parameter combination 162, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_162.json"
    BASE_NAME="param_162_rep_1"
    ;;
  163)
    echo "Running parameter combination 163, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_163.json"
    BASE_NAME="param_163_rep_1"
    ;;
  164)
    echo "Running parameter combination 164, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_164.json"
    BASE_NAME="param_164_rep_1"
    ;;
  165)
    echo "Running parameter combination 165, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_165.json"
    BASE_NAME="param_165_rep_1"
    ;;
  166)
    echo "Running parameter combination 166, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_166.json"
    BASE_NAME="param_166_rep_1"
    ;;
  167)
    echo "Running parameter combination 167, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_167.json"
    BASE_NAME="param_167_rep_1"
    ;;
  168)
    echo "Running parameter combination 168, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_168.json"
    BASE_NAME="param_168_rep_1"
    ;;
  169)
    echo "Running parameter combination 169, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_169.json"
    BASE_NAME="param_169_rep_1"
    ;;
  170)
    echo "Running parameter combination 170, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_170.json"
    BASE_NAME="param_170_rep_1"
    ;;
  171)
    echo "Running parameter combination 171, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_171.json"
    BASE_NAME="param_171_rep_1"
    ;;
  172)
    echo "Running parameter combination 172, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_172.json"
    BASE_NAME="param_172_rep_1"
    ;;
  173)
    echo "Running parameter combination 173, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_173.json"
    BASE_NAME="param_173_rep_1"
    ;;
  174)
    echo "Running parameter combination 174, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_174.json"
    BASE_NAME="param_174_rep_1"
    ;;
  175)
    echo "Running parameter combination 175, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_175.json"
    BASE_NAME="param_175_rep_1"
    ;;
  176)
    echo "Running parameter combination 176, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_176.json"
    BASE_NAME="param_176_rep_1"
    ;;
  177)
    echo "Running parameter combination 177, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_177.json"
    BASE_NAME="param_177_rep_1"
    ;;
  178)
    echo "Running parameter combination 178, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_178.json"
    BASE_NAME="param_178_rep_1"
    ;;
  179)
    echo "Running parameter combination 179, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_179.json"
    BASE_NAME="param_179_rep_1"
    ;;
  180)
    echo "Running parameter combination 180, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_180.json"
    BASE_NAME="param_180_rep_1"
    ;;
  181)
    echo "Running parameter combination 181, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_181.json"
    BASE_NAME="param_181_rep_1"
    ;;
  182)
    echo "Running parameter combination 182, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_182.json"
    BASE_NAME="param_182_rep_1"
    ;;
  183)
    echo "Running parameter combination 183, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_183.json"
    BASE_NAME="param_183_rep_1"
    ;;
  184)
    echo "Running parameter combination 184, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_184.json"
    BASE_NAME="param_184_rep_1"
    ;;
  185)
    echo "Running parameter combination 185, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_185.json"
    BASE_NAME="param_185_rep_1"
    ;;
  186)
    echo "Running parameter combination 186, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_186.json"
    BASE_NAME="param_186_rep_1"
    ;;
  187)
    echo "Running parameter combination 187, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_187.json"
    BASE_NAME="param_187_rep_1"
    ;;
  188)
    echo "Running parameter combination 188, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_188.json"
    BASE_NAME="param_188_rep_1"
    ;;
  189)
    echo "Running parameter combination 189, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_189.json"
    BASE_NAME="param_189_rep_1"
    ;;
  190)
    echo "Running parameter combination 190, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_190.json"
    BASE_NAME="param_190_rep_1"
    ;;
  191)
    echo "Running parameter combination 191, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_191.json"
    BASE_NAME="param_191_rep_1"
    ;;
  192)
    echo "Running parameter combination 192, replicate 1"
    CONFIG_FILE="5t5b/5t5b_mfcsb/configs/config_192.json"
    BASE_NAME="param_192_rep_1"
    ;;
  *)
    echo "Error: Unknown array task ID $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

# Run simulation
echo "Starting Julia simulation..."
julia run_simulation.jl --config $CONFIG_FILE --output-dir 5t5b/5t5b_mfcsb/data --base-name $BASE_NAME --timestamp --verbose --save-results

echo "Job finished at: $(date)"
