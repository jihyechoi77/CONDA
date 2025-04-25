# The First Test-time adaptation Framework for Concept Bottlenecks with Foundation Model Backbones (ICLR'25)


[![](https://img.shields.io/badge/Paper-pink?style=plastic&logo=GitBook)](https://arxiv.org/pdf/2412.14097)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This is the code for the paper:

Jihye Choi, Jayaram Raghuram, Yixuan Li, Somesh Jha. 
*CONDA: Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts*,
ICLR 2025.

* To observe the performance of feature-based predictions with foundation models (i.e., zero-shot predictions and linear probing following the feature extraction) between source vs target domains,

```angular2html
python baseline.py --out-dir "{RESULT_DIR}" \
                  --dataset {SOURCE_DATA} --dataset-shift {TARGET_DATA} \
                  --batch-size 32 \
                  --num-epochs 20 \
                  --backbone-name {BACKBONE}

```


* To compare the performance of concept-based predictions with foundation model backbones (without any adaptation) between source vs target domains,
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

* To compare the performance of concept-based predictions with foundation models (with CONDA, our test-time adaptation method) on the target domain,
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


## **📎 Reference**

If you find this code/work useful in your own research, please consider citing the following:
```bibtex
@inproceedings{
choi2025conda,
title={{CONDA}: Adaptive Concept Bottleneck for Foundation Models Under Distribution Shifts},
author={Jihye Choi and Jayaram Raghuram and Yixuan Li and Somesh Jha},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=8sfc8MwG5v}
}
```


