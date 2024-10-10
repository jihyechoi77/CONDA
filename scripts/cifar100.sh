########################
## Backbone


./scripts/corruption.sh 3

python baseline.py --out-dir "results/cifar100" \
                     --dataset "cifar100" --dataset-shift "cifar100-c-brightness-2" \
                     --batch-size 256 --num-epochs 50 \
                     --backbone-name "robustclip"


CUDA_VISIBLE_DEVICES=$1 python cbm_adapt.py --out-dir "results/$dataset" \
              --dataset "$dataset" --dataset-shift "$dataset-c-$corruption-$severity" \
              --batch-size 256 \
              --num-epochs 50 \
              --backbone-name "robustclip" \
              --concept-method "yeh" --num-concepts 170 \
              --label-ensemble \
              --lr 1e-2 \
              --adapt-cavs --adapt-classifier --rcbm --num-residual-concepts 5 \
              --adapt-steps 70 > "results/$dataset/yeh-all-$dataset-c-$corruption-$severity.txt"