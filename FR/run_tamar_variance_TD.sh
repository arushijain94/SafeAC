#!/bin/bash
Lr_pol=(1e-2)
Lr_J=(0.5)
Temp=(1.0)
Tradeoff=(0.0)
Lr_M=(0.0)
run=20
episode=1000
path_name="./"
file_name="TamarVarianceTD.py"

for temp in "${Temp[@]}"; do
  for lr_pol in "${Lr_pol[@]}"; do
    for lr_J in "${Lr_J[@]}"; do
      for tradeoff in "${Tradeoff[@]}"; do
        for lr_M in "${Lr_M[@]}"; do
          nohup python $path_name$file_name --temperature $temp --lr_P $lr_pol --lr_J $lr_J --lr_M $lr_M --mu $tradeoff --nruns $run --nepisodes $episode --seed 1 &
        done
      done
    done
  done
done

