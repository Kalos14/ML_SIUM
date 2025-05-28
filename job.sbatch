#!/bin/bash
#SBATCH --job-name=fin_407-notebook
#SBATCH --partition=fin_407
#SBATCH --account=fin/407
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=0:02:00
#SBATCH --mem=32G
#SBATCH --output=fin-407-jupyter-%j.out

module load gcc python
# Load your virtual environment or conda environment if needed
source /home/calandrino/venv/bin/activate

srun python /home/calandrino/prova.py
