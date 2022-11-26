#!/bin/bash

## Resource Request
#SBATCH --job-name=CIFAR
#SBATCH --output=CIFAR.o
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=5GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fazeelath.mohammed@student.uni-siegen.de

module load miniconda3

env_dir=$(/home/g050878/.conda/envs/augmixenv)
conda activate $(env_dir)

module load GpuModules
bash set_up.sh

python cifar.py

