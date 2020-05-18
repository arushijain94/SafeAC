#!/bin/bash
#SBATCH --account=def-dprecup            # Yoshua pays for your job
#SBATCH --mem=2G	                      # Ask for 2 GB of RAM
#SBATCH --gres=gpu:0              # Number of GPUs (per node)
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00                   # The job will run for 3 hours
#SBATCH --output=./OUT/%j-%x.out

module load python/3.6
module load cuda cudnn
source ~/.bashrc
source ~/MujocoVenv/bin/activate

temp=$1
lr_p=$2
lr_c=$3
lr_sigma=$4
eps=$5
run=$6
psi=$7
path_name="./"
file_name="Discrete_OffP.py"

python $path_name$file_name --temperature $temp  --lr_theta $lr_p --lr_critic $lr_c --lr_sigma $lr_sigma --nepisodes $eps --nruns $run --psi $psi
