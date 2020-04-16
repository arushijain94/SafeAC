#!/bin/bash
Lr_pol=(1e-3)
Lr_J=(0.05)
Temp=(1.0)
Tradeoff=(1e-4 1e-3 5e-3)
Lr_M=(0.01 0.0025 5e-3)
Gamma=(0.99)
path_name="./"
file_name="VarianceAdjustedAC.py"

for temp in "${Temp[@]}"; do
  for lr_pol in "${Lr_pol[@]}"; do
    for lr_J in "${Lr_J[@]}"; do
      for tradeoff in "${Tradeoff[@]}"; do
        for gamma in "${Gamma[@]}"; do
          for lr_M in "${Lr_M[@]}"; do
            nohup python $path_name$file_name --temperature $temp --lr_pol $lr_pol --lr_J $lr_J --lr_M $lr_M --tradeoff $tradeoff --gamma $gamma --seed 20 &
          done
        done
      done
    done
  done
done
