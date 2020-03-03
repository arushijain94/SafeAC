#!/bin/bash
Lr_p=(1e-3)
Lr_c=(1e-1)
Lr_sigma=(1e-1)
Lam=(0.4)
Temp=(5e-2)
PsiFixed=("True")
PsiRate=(1)
Psi=(0.0)
Seed=(10)
Episodes=(100)

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
				                        nohup python SAC_trace_OnP.py --temperature $temp  --lr_theta $lr_p --lr_critic $lr_c --lr_sigma $lr_sigma --lmbda $lam  --nepisodes $eps --psi $psi --psiFixed $psifixed --psiRate $psirate --seed $seed &
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
