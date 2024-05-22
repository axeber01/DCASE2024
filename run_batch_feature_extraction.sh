#!/bin/bash
#SBATCH --account=Berzelius-2023-349
#SBATCH --gpus 1
#SBATCH -t 2-00:00:00

# job name
#SBATCH -J Train_resnet_1
#
# Remap stdout and stderr to write to these files
#SBATCH -o Train_resnet_%A_%a.out
#SBATCH -e Train_resnet_%A_%a.out

module load Anaconda/2023.09-0-hpc1-bdist
conda activate dcase
python batch_feature_extraction.py 6
