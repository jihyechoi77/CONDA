# Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts

This is the anonymized repository for the submission: Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts.


* To observe the performance of feature-based predictions with foundation models (i.e., zero-shot predictions and linear probing following the feature extraction) between source vs target domains,

```angular2html
python baseline.py --out-dir "{RESULT_DIR}" \
                  --dataset {SOURCE_DATA} --dataset-shift {TARGET_DATA} \
                  --batch-size 32 \
                  --num-epochs 20 \
                  --backbone-name {BACKBONE}

```


* To compare the performance of concept-based predictions with foundation models (without any adaptation) between source vs target domains,
```angular2html
# first construct the concept bottleneck
python3 learn_concepts_dataset.py --dataset-name={SOURCE_DATA} --backbone-name={BACKBONE} \
--out-dir="datasets/concepts/"

# then you are ready to try concept-based predictions!
python cbm.py --out-dir "results/waterbirds" \
              --dataset {SOURCE_DATA} --dataset-shift {TARGET_DATA} \
              --batch-size 32 \
              --num-epochs 20 \
              --backbone-name {BACKBONE}

```

* To compare the performance of concept-based predictions with foundation models (with CONDA, our **CON**cept-based **D**ynamic **A**daptation) between source vs target domains,

```angular2html
python cbm_adapt.py --out-dir "{RESULT_DIR}" \
              --dataset {SOURCE_DATA} --dataset-shift {TARGET_DATA} \
              --batch-size 32 \
              --num-epochs 20 \
              --backbone-name {BACKBONE} \
              --concept-method "pcbm" \
              --label-ensemble \
              --adapt-steps 20 \
              --lr 1e-1 \
              --adapt-cavs \
              --adapt-classifier \
              --rcbm --num-residual-concepts 5
```
Each flag of `--adapt-cavs`, `--adapt-classifier`, `--rcbm` turns on each component of CONDA; 1) Concept Score Alignment, 2) Linear Probing Adaptation, and 3) Residual Concept Bottleneck, respectively.

The `scripts/` directory contains scripts to run experiments for each dataset setting.