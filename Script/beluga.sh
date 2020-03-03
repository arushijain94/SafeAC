#!/bin/bash

cp -n -r ~/projects/rpp-bengioy/ajan25/SafeActorCritic/SafeAC_ICML2019/FR ./
mkdir -p ./OUT

# Envs: FR
Lr_p=(1e-3)
Lr_c=(1e-1)
Lr_sigma=(1e-2)
Lam=(0.6 0.8)
Temp=(5e-2 1e-3)
PsiFixed=("True")
PsiRate=(1)
Psi=(0.0)
Seed=(10)
Episodes=(2000)

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
				                        sbatch ~/scratch/run_fr.sh $temp $lr_p $lr_c $lr_sigma $lam $eps $psi $psifixed $psirate $seed &
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

