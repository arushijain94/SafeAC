#!/bin/bash
mkdir -p ./OUT

# 'Ant-v2' 'HalfCheetah-v2' 'Hopper-v2' 'Walker2d-v2', 'HumanoidStandup-v2', 'Humanoid-v2'
# Runnning all 20 seeds for one type of environment


envs=('HumanoidStandup-v2')
eps=(200000)
batch=(20)
psis=(0.15)

for env in "${envs[@]}"
do
	for seed in {1..20}
	do
		for psi in "${psis[@]}"
		do
			sbatch ./run_mujoco.sh $env $psi $eps $batch $seed
		done
	done
done




