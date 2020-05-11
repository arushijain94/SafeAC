#!/bin/bash
Lr_pol=(1e-2)
Lr_J=(0.1 0.05)
Temp=(1.)
Tradeoff=(0.0)
Lr_M=(0.0)
Runs=(100)
Episodes=(6000)
path_name="./"
file_name="TamarVarianceMC.py"

for temp in "${Temp[@]}"; do
  for lr_pol in "${Lr_pol[@]}"; do
    for lr_J in "${Lr_J[@]}"; do
      for tradeoff in "${Tradeoff[@]}"; do
        for run in "${Runs[@]}"; do
          for lr_M in "${Lr_M[@]}"; do
            for eps in "${Episodes[@]}"; do
              nohup python $path_name$file_name --temperature $temp --lr_pol $lr_pol --lr_J $lr_J --lr_M $lr_M --tradeoff $tradeoff --nruns $run --nepisodes $eps --seed 50 &
            done
          done
        done
      done
    done
  done
done
