python baseline.py --out-dir "results/metashift" \
                  --dataset "metashift" --dataset-shift "metashift_shift" \
                  --batch-size 32 \
                  --num-epochs 20 \
                  --backbone-name "clip:ViT-L-14"


########################

# prepare static concept bank
python3 learn_concepts_dataset.py --dataset-name="metashift" --backbone-name="clip:ViT-L-14" \
--out-dir="datasets/concepts/" --seed=0


# 1) concept bottleneck alignment
python cbm_adapt.py --out-dir "results/metashift" \
              --dataset "metashift" --dataset-shift "metashift_shift" \
              --batch-size 32 \
              --num-epochs 20 \
              --backbone-name "clip:ViT-L-14" \
              --concept-method "pcbm" \
              --label-ensemble \
              --adapt-cavs --lr 1e-2


# 2) + classifier adjustment
python cbm_adapt.py --out-dir "results/metashift" \
              --dataset "metashift" --dataset-shift "metashift_shift" \
              --batch-size 32 \
              --num-epochs 20 \
              --backbone-name "clip:ViT-L-14" \
              --concept-method "pcbm" \
              --label-ensemble \
              --adapt-classifier --lr 1e-1 \
              --adapt-steps 50


# 3) residual concept bottleneck
python cbm_adapt.py --out-dir "results/metashift" \
              --dataset "metashift" --dataset-shift "metashift_shift" \
              --batch-size 32 \
              --num-epochs 20 \
              --backbone-name "clip:ViT-L-14" \
              --concept-method "pcbm" \
              --label-ensemble \
              --adapt-steps 50 \
              --rcbm --lr 1e-1 \
              --num-residual-concepts 5

# 4) all
python cbm_adapt.py --out-dir "results/metashift" \
              --dataset "metashift" --dataset-shift "metashift_shift" \
              --batch-size 32 \
              --num-epochs 20 \
              --backbone-name "clip:ViT-L-14" \
              --concept-method "pcbm" \
              --label-ensemble \
              --adapt-steps 50 \
              --adapt-cavs --adapt-classifier \
              --rcbm --lr 1e-1 \
              --num-residual-concepts 5
