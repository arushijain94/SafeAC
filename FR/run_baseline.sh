#!/bin/bash
Lr_p=(1e-3)
Lr_c=(1e-1)
Lam=(0.4)
Temp=(5e-2)
B=(1000 2000 3000 4000 5500 6000)
Seed=(10)
Episodes=(800)

for temp in "${Temp[@]}"; do
	for lr_p in "${Lr_p[@]}"; do
		for lr_c in "${Lr_c[@]}"; do
			for b in "${B[@]}"; do
				for lam in "${Lam[@]}"; do
				    for eps in "${Episodes[@]}"; do
					    nohup python VarianceBaseline.py --temperature $temp  --lr_theta $lr_p --lr_critic $lr_c --b $b --lam $lam  --nepisodes $eps &
					done
				done
			done
		done
	done
done
