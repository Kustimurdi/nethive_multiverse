#!/bin/bash
# Submit job array for bee simulation parameter sweep

echo "Submitting job array: job_array.slurm"
sbatch single_tasks/new_tasks_training/job_array.slurm

echo "Job array submitted!"
echo "Check status with: squeue -u $USER"
echo "Monitor progress in: single_tasks/new_tasks_training/logs/"
