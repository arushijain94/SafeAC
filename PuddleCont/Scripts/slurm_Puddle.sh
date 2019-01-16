#!/bin/bash

#SBATCH --mem=10G
#SBATCH --time=5-00:00:00

hostname
whoami
cd $HOME/SAC/Puddle/ICLR_2019/OffPolicy_PuddleWorld
python Puddle_OffPolicy.py --nepisodes 3500 --psi 0.05 --lr_critic 5e-3 --lr_theta 5e-5 --lr_sigma 5e-4 --seed 10 --nruns 50
echo “DONE
