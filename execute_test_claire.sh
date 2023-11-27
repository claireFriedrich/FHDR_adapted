#!/bin/bash
#SBATCH --chdir /scratch/izar/cfriedri
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --mem 50G

echo STARTING AT `date`

cd /home/cfriedri/FHDR_adapted
echo SUCCESSFULLY CHANGED LOCATION

python3 -u  test.py --ckpt_path /path/to/pth/checkpoint

echo FINISHED at `date`

