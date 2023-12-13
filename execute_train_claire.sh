#!/bin/bash
#SBATCH --chdir /scratch/izar/cfriedri
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=8:0:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem 1G

echo STARTING AT `date`
nvidia-smi

cd /home/cfriedri/FHDR_adapted
echo SUCCESSFULLY CHANGED LOCATION

python3 -u train.py 

echo FINISHED at `date`