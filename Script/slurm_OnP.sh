#!/bin/bash

#SBATCH --mem=5G
#SBATCH --time=1-00:00:00

hostname
whoami
cd $HOME/SAC
python SAC_trace_OnP.py --psi 0.1 --lr_critic 0.1 --temperature 0.05 --nruns 50 --seed 20 --lr_sigma 0.02 --nepisodes 5000
echo “DONE
