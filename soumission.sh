#!/bin/bash
# Submission script for Hercules2
#SBATCH --job-name=MachineLearnSD
#SBATCH --time=05:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:4"
#SBATCH --mem-per-cpu=1000 # megabytes
#SBATCH --partition=gpu
#
#SBATCH --mail-user=jarod.ketchakouakep@student.unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --output=outfile

module purge 
ml  releases/2022b TensorFlow/2.13.0-foss-2022b
ml Python
ml matplotlib


echo "Task_ID : $SLURM_ARRAY_TASK_ID"
python TrialAndError.py $SLURM_ARRAY_TASK_ID
