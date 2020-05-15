#!/bin/bash
Lr_p=(1e-5)
Lr_c=(1e-3 1e-2)
Lr_sigma=(0.0)
Lam=(1.0)
Temp=(0.001)
Psi=(0.0)
seed=1
run=2
eps=500
path_name="./"
file_name="Cont_OffP.py"

for temp in "${Temp[@]}"; do
	for lr_p in "${Lr_p[@]}"; do
		for lr_c in "${Lr_c[@]}"; do
			for psi in "${Psi[@]}"; do
				for lam in "${Lam[@]}"; do
	                for lr_sigma in "${Lr_sigma[@]}"; do
	                    nohup python $path_name$file_name --temperature $temp  --lr_theta $lr_p --lr_critic $lr_c --lr_sigma $lr_sigma --lmbda $lam  --nepisodes $eps --nruns $run --psi $psi --seed $seed &
            		done
				done
			done
		done
	done
done

