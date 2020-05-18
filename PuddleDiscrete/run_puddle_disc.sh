#!/bin/bash
Lr_p=(0.05)
Lr_c=(0.5)
Lr_sigma=(0.5 0.25)
Temp=(100)
Psi=(1e-3 5e-3 0.075)
seed=1
run=20
eps=500
path_name="./"
file_name="Discrete_OffP.py"

for temp in "${Temp[@]}"; do
	for lr_p in "${Lr_p[@]}"; do
		for lr_c in "${Lr_c[@]}"; do
			for psi in "${Psi[@]}"; do
        for lr_sigma in "${Lr_sigma[@]}"; do
          nohup python $path_name$file_name --temperature $temp  --lr_theta $lr_p --lr_critic $lr_c --lr_sigma $lr_sigma --nepisodes $eps --nruns $run --psi $psi --seed $seed &
        done
			done
		done
	done
done

