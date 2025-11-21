#!/bin/bash
# Submit job array for bee simulation parameter sweep

echo "Submitting job array: job_array.slurm"
sbatch 2t2b_high_deadtime/mnist_fashion_mnist/job_array.slurm

echo "Job array submitted!"
echo "Check status with: squeue -u $USER"
echo "Monitor progress in: 2t2b_high_deadtime/mnist_fashion_mnist/logs/"
