#!/bin/bash
#SBATCH --job-name=test-job
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node v100:1
#SBATCH --mem-per-gpu=8G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=suchilu@iu.edu
#SBATCH -A r00130
module purge
module load deeplearning/2.12.0
nvidia-smi
pip3 install transformers datasets evaluate seqeval
python fine_tuning_nlp.py
