#!/bin/bash
mkdir -p ./OUT

# 'Ant-v2' 'HalfCheetah-v2' 'Hopper-v2' 'Walker2d-v2', 'HumanoidStandup-v2', 'Humanoid-v2'
# 'Ant' 'HCheet' 'Hopper' 'Walker', 'HumanoidStand', 'Humanoid'
# Runnning all 20 seeds for one type of environment
# HC: batch 5
# Humanoid: batch

envs=('HalfCheetah-v2')
eps=(15000)
batch=(5)
psis=(0.125 0.05 0.15)
jobname='HC'

for env in "${envs[@]}"
do
	for seed in {1..20}
	do
		for psi in "${psis[@]}"
		do
			sbatch -J $jobname$psi ./run_mujoco.sh $env $psi $eps $batch $seed
		done
	done
done




