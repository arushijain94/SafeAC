#!/bin/bash
mkdir -p ./OUT

# 'Ant-v2' 'HalfCheetah-v2' 'Hopper-v2' 'Walker2d-v2', 'HumanoidStandup-v2', 'Humanoid-v2'
# Runnning all 20 seeds for one type of environment
# HC: batch 5
# Humanoid: batch

envs=('Walker2d-v2')
eps=(40000)
batch=(10)
psis=(0.005)
jobname='Walker'
seeds=(9 13 10)

for env in "${envs[@]}"
do
	for seed in "${seeds[@]}"
	do
		for psi in "${psis[@]}"
		do
			sbatch -J $jobname$psi ./run_mujoco.sh $env $psi $eps $batch $seed
		done
	done
done




