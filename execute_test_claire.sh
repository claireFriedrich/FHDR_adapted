#!/bin/bash
#SBATCH --chdir /scratch/izar/cfriedri
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=8:0:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem 16G

echo STARTING AT `date`

cd /home/cfriedri/FHDR_adapted
echo SUCCESSFULLY CHANGED LOCATION

python3 -u  test.py --ckpt_path "/home/cfriedri/FHDR_adapted/FHDR-iter-2.ckpt"

echo FINISHED at `date`

