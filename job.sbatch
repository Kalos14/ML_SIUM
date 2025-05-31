#!/bin/bash
#SBATCH --job-name=fin-407-notebook
#SBATCH --qos=fin-407
#SBATCH --account=fin-407
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=fin-407-jupyter-%j.out

module load gcc python
# Load your virtual environment or conda environment if needed
source /home/$USER/venv/bin/activate

srun python /home/$USER/ML_SIUM/test_supercomputer_PCA_mom.py
