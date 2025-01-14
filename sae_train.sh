#!/bin/bash -l

#SBATCH -J trainsae
#SBATCH -p gpu                
#SBATCH -t 10:00:00             
#SBATCH -N 1   
#SBATCH -C v100
#SBATCH --gpus=1
#SBATCH --mem=50G                 
#SBATCH --output=saetrain.out

module load python
module load gcc
module load cuda
module load cudnn
source /mnt/home/rzhang/envs/interp/bin/activate

srun python sae_train.py
