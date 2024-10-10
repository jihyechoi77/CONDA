########################

python baseline.py --out-dir "results/waterbirds" \
                  --dataset "waterbirds" --dataset-shift "waterbirds_shift" \
                  --batch-size 32 \
                  --num-epochs 20 \
                  --backbone-name "clip:ViT-L-14"


########################

# prepare static concept bank
python3 learn_concepts_dataset.py --dataset-name="waterbirds" --backbone-name="clip:ViT-L-14" \
--out-dir="datasets/concepts/" --seed=0

python cbm_adapt.py --out-dir "results/waterbirds" \
              --dataset "waterbirds" --dataset-shift "waterbirds_shift" \
              --batch-size 32 \
              --num-epochs 20 \
              --backbone-name "clip:ViT-L-14" \
              --concept-method "pcbm" \
              --label-ensemble \
              --adapt-steps 20 \
              --lr 1e-1 \
              --adapt-cavs --adapt-classifier --rcbm --num-residual-concepts 5




