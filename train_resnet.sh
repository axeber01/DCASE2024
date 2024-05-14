#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 2-00:00:00
 
module load Anaconda/2023.09-0-hpc1-bdist 
conda activate dcase
python train_resnet_seldformat.py
