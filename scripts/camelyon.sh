########################
## Backbone

python baseline.py --out-dir "results/camelyon17" \
                  --dataset "camelyon17" --dataset-shift "camelyon17-shift" \
                  --batch-size 64 \
                  --num-epochs 50 \
                  --backbone-name "medclip"


########################
# 1) PCBM

# prepare static concept bank
python3 learn_concepts_dataset.py --dataset-name="camelyon17" --backbone-name="medclip" \
--out-dir="datasets/concepts/" --seed=0



### PCBM (add --ensemble to use ensembling for pseudo-labels)
# 1) concept bottleneck alignment
python cbm_adapt.py --out-dir "results/camelyon17" \
              --dataset "camelyon17" --dataset-shift "camelyon17-shift" \
              --batch-size 64 \
              --num-epochs 20 \
              --backbone-name "medclip" \
              --concept-method "pcbm" \
              --label-ensemble \
              --adapt-classifier --lr 1e-2 \
              --adapt-steps 20


python cbm_adapt.py --out-dir "results/camelyon17" \
              --dataset "camelyon17" --dataset-shift "camelyon17-shift" \
              --batch-size 64 \
              --num-epochs 20 \
              --backbone-name "medclip" \
              --concept-method "yeh" --num-concepts 50 \
              --label-ensemble \
              --adapt-classifier --lr 1e-2 \
              --adapt-steps 20



# 2) + classifier adjustment
python cbm_adapt.py --out-dir "results/waterbirds" \
              --dataset "waterbirds" --dataset-shift "waterbirds_shift" \
              --batch-size 32 \
              --num-epochs 20 \
              --backbone-name "clip:ViT-L-14" \
              --concept-method "pcbm" \
              --label-ensemble \
              --adapt-classifier --lr 1e-1 \
              --adapt-steps 20





# 4) all
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

### Yeh et al. (add --ensemble to use ensembling for pseudo-labels)
# 1) concept bottleneck alignment
python cbm_adapt.py --out-dir "results/waterbirds" \
              --dataset "waterbirds" --dataset-shift "waterbirds_shift" \
              --batch-size 32 \
              --num-epochs 20 \
              --backbone-name "clip:ViT-L-14" \
              --concept-method "yeh" \
              --num-concepts 85 \
              --label-ensemble \
              --lr 1e-2 \
              --adapt-steps 50 \
              --adapt-cavs --adapt-classifier --rcbm --num-residual-concepts 5

