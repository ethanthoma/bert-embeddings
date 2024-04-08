#!/usr/bin/env bash

#SBATCH --account=def-gerope
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --output=output/%j.out
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1

# Load modules for environment
module load StdEnv/2023
module load python/3.11
module load scipy-stack

source ENV/bin/activate

# run eval
python app/__init__.py

deactivate
