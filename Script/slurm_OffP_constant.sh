#!/bin/bash

#SBATCH --mem=3G
#SBATCH --time=1-00:00:00

hostname
whoami
cd $HOME/SAC
python SAC_trace_OffP_new_constant.py --nepisodes 2000 --psi 0.15 --kstep 40 --lr_critic 25e-3 --lr_theta 25e-6 --lr_sigma 5e-5 --seed 10 --temperature 0.25 --lmbda 0.5 --nruns 50
echo “DONE
