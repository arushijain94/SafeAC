#!/bin/bash
Lr_pol=(1e-3)
Lr_J=(1e-1)
Temp=(5e-2 1e-3)
Tradeoff=(0.05 0.1)
Seed=(10)
Episodes=(2000)
path_name="./"
file_name="VarianceAdjustedAC.py"

for temp in "${Temp[@]}"; do
  for lr_pol in "${Lr_pol[@]}"; do
    for lr_J in "${Lr_J[@]}"; do
      for tradeoff in "${Tradeoff[@]}"; do
        for eps in "${Episodes[@]}"; do
          for seed in "${Seed[@]}"; do
            nohup python $path_name$file_name --temperature $temp --lr_pol $lr_pol --lr_J $lr_J --lr_M $lr_J --nepisodes $eps --tradeoff $tradeoff --seed $seed &
          done
        done
      done
    done
  done
done
