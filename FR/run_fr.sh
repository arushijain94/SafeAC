#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --mem=2G	                      # Ask for 2 GB of RAM
#SBATCH --gres=gpu:0              # Number of GPUs (per node)
#SBATCH --cpus-per-task=1
#SBATCH --time=7:00:00                   # The job will run for 7 hours
#SBATCH --output=./OUT/ajain-%j.out
#SBATCH --mail-user=arushi.jain@mail.mcgill.ca
#SBATCH --mail-type=FAIL

# 1. Create your environement locally
module load python/3.6
source ~/SafeFR/bin/activate

#Env: FR environment
temp=$1
lr_p=$2
lr_c=$3
lr_sigma=$4
lam=$5
eps=$6
psi=$7
psifixed=$8
psirate=$9
seed=$10

python ./SAC_OnP_Direct.py --temperature $temp  --lr_theta $lr_p --lr_critic $lr_c --lr_sigma $lr_sigma --lmbda $lam  --nepisodes $eps --psi $psi --psiFixed $psifixed --psiRate $psirate --seed $seed
