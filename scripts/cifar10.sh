########################
## Backbone


./scripts/corruption.sh cifar10


########################
# prepare static concept bank
python3 learn_concepts_dataset.py --dataset-name="cifar10" --backbone-name="robustclip" \
--out-dir="datasets/concepts/" --seed=0

# 1) concept bottleneck alignment
python cbm_adapt.py --out-dir "results/cifar10" \
              --dataset "cifar10" --dataset-shift "cifar10-c-brightness-2" \
              --batch-size 128 \
              --num-epochs 20 \
              --backbone-name "robustclip" \
              --concept-method "pcbm" \
              --label-ensemble \
              --lr 1e-2 --adapt-cavs \
              --adapt-steps 70


python cbm_adapt.py --out-dir "results/cifar10" \
              --dataset "cifar10" --dataset-shift "cifar10-c-gaussian_noise-2" \
              --batch-size 128 \
              --num-epochs 20 \
              --backbone-name "robustclip" \
              --concept-method "pcbm" \
              --label-ensemble \
              --lr 1e-1 \
              --rcbm --num-residual-concepts 10 \
              --adapt-steps 100


python cbm_adapt.py --out-dir "results/cifar10" \
              --dataset "cifar10" --dataset-shift "cifar10-c-gaussian_noise-2" \
              --batch-size 128 \
              --num-epochs 20 \
              --backbone-name "robustclip" \
              --concept-method "yeh" --num-concepts 170 \
              --label-ensemble \
              --adapt-cavs --lr 1e-2 \
              --adapt-steps 70


