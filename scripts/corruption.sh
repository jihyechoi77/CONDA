#!/bin/bash
gpu=$1


#dataset=cifar10
#bs=128

dataset=cifar100
bs=256


severity=2
backbone=robustclip

# Define the array of corruptions
declare -a corruptions=("gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" "glass_blur" "motion_blur" "zoom_blur" "snow" "frost" "fog" "brightness" "contrast" "elastic_transform" "pixelate" "jpeg_compression")

# Loop through the array and run the command with each corruption
for corruption in "${corruptions[@]}"; do
  dataset_shift=$dataset-c-$corruption-$severity



  echo "Running experiment with corruption: $corruption"
  CUDA_VISIBLE_DEVICES=$1 python baseline.py --out-dir "results/$dataset" \
                     --dataset "$dataset" --dataset-shift "$dataset_shift" \
                     --batch-size $bs --num-epochs 50 \
                     --backbone-name "$backbone"  > "results/$dataset/baseline_2nn-$dataset_shift.txt"

#  CUDA_VISIBLE_DEVICES=$1 python cbm_adapt.py --out-dir "results/$dataset" \
#              --dataset "$dataset" --dataset-shift "$dataset-c-$corruption-$severity" \
#              --batch-size $bs \
#              --num-epochs 20 \
#              --backbone-name "robustclip" \
#              --concept-method "yeh" --num-concepts 170 \
#              --label-ensemble \
#              --lr 1e-1 \
#              --adapt-cavs \
#              --adapt-steps 50 > "results/$dataset/yeh-csa-$dataset-c-$corruption-$severity.txt"
  done

echo "All experiments completed."