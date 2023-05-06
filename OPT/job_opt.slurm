#!/bin/bash
#SBATCH --job-name=test-job
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node v100:2
#SBATCH --mem-per-gpu=128G
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abhiasth@iu.edu
#SBATCH -A r00134
module purge
module load deeplearning/2.12.0
nvidia-smi
pip3 install transformers datasets
python GPT-neo-all.py
