#!/bin/bash
Lr_p=(5e-2)
Lr_c=(0.5)
Lr_sigma=(0.075)
Lam=(0.6)
Temp=(0.5)
PsiFixed=("True")
PsiRate=(400)
Psi=(0.05 0.1 0.25)
Seed=(1)
run=5
Episodes=(500)
path_name="./"
file_name="SAC_trace_OnP.py"

for temp in "${Temp[@]}"; do
	for lr_p in "${Lr_p[@]}"; do
		for lr_c in "${Lr_c[@]}"; do
			for psi in "${Psi[@]}"; do
				for lam in "${Lam[@]}"; do
				    for eps in "${Episodes[@]}"; do
				        for psifixed in "${PsiFixed[@]}"; do
				            for psirate in "${PsiRate[@]}"; do
				                for lr_sigma in "${Lr_sigma[@]}"; do
				                    for seed in "${Seed[@]}"; do
				                        nohup python $path_name$file_name --temperature $temp  --lr_theta $lr_p --lr_critic $lr_c --lr_sigma $lr_sigma --lmbda $lam  --nepisodes $eps --nruns $run --psi $psi --psiFixed $psifixed --psiRate $psirate --seed $seed &
				                    done
                        		done
				            done
				        done
					done
				done
			done
		done
	done
done

