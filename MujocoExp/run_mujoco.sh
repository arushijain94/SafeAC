#!/bin/bash
#SBATCH --account=rrg-bengioy-ad            # Yoshua pays for your job
#SBATCH --mem=8G	                      # Ask for 2 GB of RAM
#SBATCH --gres=gpu:0              # Number of GPUs (per node)
#SBATCH --cpus-per-task=1
#SBATCH --time=0-23:00:00                   # The job will run for 3 hours
#SBATCH --output=./OUT/sac-%j.out
#SBATCH --mail-user=arushijain130@gmail.com
#SBATCH --mail-type=FAIL

module load python/3.6
module load cuda cudnn
source ~/MujocoVenv/bin/activate

env=$1
psi=$2
eps=$3
batch=$4
seed=$5
path_name="./"
file_name="train.py"

python $path_name$file_name $env -n $eps -b $batch -p $psi -s $seed