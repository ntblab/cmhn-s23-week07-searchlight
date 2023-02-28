#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=searchlight-%j.out
#SBATCH --job-name searchlight
#SBATCH -p day 
#SBATCH --reservation=cmhn
#SBATCH -t 1:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH -n 2

# Set up the environment
module load miniconda
module load OpenMPI
conda activate /gpfs/gibbs/project/cmhn/share/conda_envs/mybrainiak

# Make script executable
chmod 700 ./searchlight.py

# Run the python script
srun --mpi=pmi2 ./searchlight.py
