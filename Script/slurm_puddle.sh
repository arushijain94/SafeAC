#!/bin/bash

#SBATCH --mem=3G
#SBATCH --time=1-00:00:00

hostname
whoami
cd $HOME/SAC
python Puddle_discrete_OffP.py --nepisodes 4000 --psi 0.0 --kstep 40 --lr_critic 0.075 --lr_theta 5e-4 --lr_sigma 5e-4 --seed 10 --temperature 0.75 --lmbda 0.5 --nruns 50
echo “DONE
