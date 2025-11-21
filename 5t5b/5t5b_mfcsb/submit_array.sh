#!/bin/bash
# Submit job array for bee simulation parameter sweep

echo "Submitting job array: job_array.slurm"
sbatch 5t5b/5t5b_mfcsb/job_array.slurm

echo "Job array submitted!"
echo "Check status with: squeue -u $USER"
echo "Monitor progress in: 5t5b/5t5b_mfcsb/logs/"
