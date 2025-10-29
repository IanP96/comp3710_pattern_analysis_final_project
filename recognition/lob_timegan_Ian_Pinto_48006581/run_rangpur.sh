#!/bin/bash

# script to run training on UQ Rangpur

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --job-name=final_proj

conda init
conda activate comp3710
python train.py --env rangpur > output/stdout.txt 2> output/stderr.txt
python predict.py --env rangpur