# Model Folding Stress Test

POSTECH Final Project, Spring 2026  
**Track 2** — stress-testing a published paper with public code.

## Paper

**Forget the Data and Fine-Tuning! Just Fold the Network to Compress.**  
Dong Wang, Haris Šikić, Lothar Thiele, Olga Saukh. *ICLR 2025.*  
[arXiv:2502.10216](https://arxiv.org/abs/2502.10216)

## TL;DR

Model folding's data-free variance repair (Fold-AR) closely tracks the data-driven upper bound (Fold-R) for layer-wise sparsity below ~35%, but breaks down sharply between 45–75% sparsity, where the gap to Fold-R grows by ~20 points. We also document four compatibility patches needed to run the released code on a current Python 3.12 / PyTorch 2.x environment.

## Main result

Test accuracy on ResNet18/CIFAR10 as a function of layer-wise sparsity.

| Sparsity | Fold-Naive | **Fold-AR** | Fold-R |
|---:|---:|---:|---:|
|  1%  | 94.49 | 94.58 | 94.54 |
| 10%  | 93.07 | 93.54 | 93.99 |
| 25%  | 84.42 | 90.97 | 92.16 |
| 35%  | 69.16 | 81.86 | 89.31 |
| **45%** | 28.47 | 70.65 | **84.34** |
| **55%** | 18.41 | **44.29** | 71.44 |
| 65%  | 12.83 | 24.25 | 56.10 |
| 75%  | 10.76 | 12.14 | 28.84 |
| 85%  | 10.17 |  9.63 | 14.79 |
| 95%  | 10.00 | 10.39 | 10.09 |

Original model: 94.57% (ResNet18 trained from scratch, 60 epochs).  
Random chance for CIFAR10 is 10%.

## Compatibility patches applied

The released code (https://github.com/marza96/ModelFolding) targets Python 3.8 / older PyTorch. To run on Colab (Python 3.12, PyTorch 2.x), we applied:

| # | Where | Change |
|---|---|---|
| 1 | `utils/weight_clustering.py:10` | `from hkmeans import HKMeans` → `from sklearn.cluster import KMeans as HKMeans`. `hartigan-kmeans` fails to build on Python 3.12 because its `versioneer.py` calls the removed `configparser.SafeConfigParser`. |
| 2 | `utils/weight_clustering.py:38` | dropped `n_jobs=-1` (not accepted by sklearn KMeans), set `verbose=0`. |
| 3 | `utils/utils.py:149` | `except torch.nn.modules.module.ModuleAttributeError` → `except AttributeError` (class removed in PyTorch 2.x). |
| 4 | (install) | `pip install thop` — used in main script but missing from `requirements.txt`. |
| 5 | (checkpoint) | the recommended `PyTorch_CIFAR10` checkpoint has a different forward graph (initial `maxpool`, single ReLU per block) from the folding repo's ResNet18. We instead trained the folding-repo ResNet18 from scratch on CIFAR10 → 94.57% test acc. |

## How to reproduce

The full pipeline runs in a Colab notebook with a T4 GPU. Estimated end-to-end runtime: ~2.5 hours (60-epoch training + three folding sweeps).

```bash
# 1) Clone
git clone https://github.com/marza96/ModelFolding
cd ModelFolding

# 2) Install
pip install torch torchvision thop scikit-learn tqdm wandb

# 3) Apply patches (or use the patched files in patches/)

# 4) Train ResNet18 on CIFAR10 (60 epochs, ~40 min on a T4)
#    see notebooks/train_resnet18.ipynb

# 5) Run all three REPAIR variants (sweep over 14 sparsity points each)
export WANDB_MODE=offline
python resnet18_cifar10_weight_clustering.py \
    --checkpoint resnet18_cifar10_marza96.pt \
    --repair "DF_REPAIR" \
    --proj_name stress_test --exp_name baseline_fold_ar
python resnet18_cifar10_weight_clustering.py \
    --checkpoint resnet18_cifar10_marza96.pt \
    --repair "NO_REPAIR" \
    --proj_name stress_test --exp_name baseline_fold_naive
python resnet18_cifar10_weight_clustering.py \
    --checkpoint resnet18_cifar10_marza96.pt \
    --repair "REPAIR" \
    --proj_name stress_test --exp_name baseline_fold_r
```

## Repository structure

```
.
├── README.md                  — this file
├── report/
│   └── final_report.pdf       — NeurIPS-formatted 6-page report
├── results/
│   ├── baseline_results.csv   — sparsity × accuracy for all three variants
│   ├── fig1_baseline_comparison.png
│   └── fig2_ar_breakdown.png

```

## Citation

If you reference this stress test, please also cite the original paper:

```bibtex
@inproceedings{wang2025forget,
  title     = {Forget the Data and Fine-tuning! Just Fold the Network to Compress},
  author    = {Wang, Dong and \v{S}iki\'{c}, Haris and Thiele, Lothar and Saukh, Olga},
  booktitle = {ICLR},
  year      = {2025}
}
```
