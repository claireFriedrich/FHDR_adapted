#!/bin/bash
#SBATCH --chdir /scratch/izar/ckalberm
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --mem 50G

echo STARTING AT `date`

cd /home/ckalberm/FHDR_adapted
echo SUCCESSFULLY CHANGED LOCATION

#python3 -u  test.py --ckpt_path /home/ckalberm/FHDR_adapted/checkpoints
python3 -u  test.py 

echo FINISHED at `date`