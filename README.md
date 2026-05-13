# Model Folding Stress Test

POSTECH Final Project, Spring 2026  
Track 2: Stress-testing a published paper with public code.

**Author**: Donggeon Kim (20242752)

## Paper

**Forget the Data and Fine-tuning! Just Fold the Network to Compress** (ICLR 2025)  
Wang, Šikić, Thiele, Saukh

## TL;DR

Model folding's data-free variance repair (Fold-AR) closely tracks the data-driven upper bound (Fold-R) for layer-wise sparsity below ~35%, but breaks down sharply between 45-75% sparsity, where the gap to Fold-R grows by ~20 points. I also document four compatibility patches needed to run the released code on a current Python 3.12 / PyTorch 2.x environment.

## Status

- Done: ResNet18 / CIFAR10 baseline, three REPAIR variants (Fold-Naive, Fold-AR, Fold-R), 14 sparsity points each
- Attempted but did not finish: VGG11-BN extension. Training worked but the sweep ran more than 24 hours on T4 without producing a result. See report Section 5.

## Main result (ResNet18 / CIFAR10)

Test accuracy at selected layer-wise sparsities (original model: 94.57%, random chance: 10%):

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

Full sweep in `results/baseline_results.csv`.

## Compatibility patches

The released code (https://github.com/marza96/ModelFolding @ commit 33e2818) targets Python 3.8 / older PyTorch. To run on current Colab (Python 3.12, PyTorch 2.x), I applied four patches:

| # | Where | Change | Reason |
|---|---|---|---|
| 1 | `utils/weight_clustering.py:10` | `from hkmeans import HKMeans` → `from sklearn.cluster import KMeans as HKMeans` | `hartigan-kmeans` fails to build on Python 3.12 (its `versioneer.py` uses `configparser.SafeConfigParser`, which was removed). |
| 2 | `utils/weight_clustering.py:38` | Dropped `n_jobs=-1`, set `verbose=0` | Not accepted by sklearn KMeans. |
| 3 | `utils/utils.py:149` | `except torch.nn.modules.module.ModuleAttributeError` → `except AttributeError` | The class was removed in PyTorch 2.x. |
| 4 | (install) | `pip install thop` | Imported by the main script but missing from `requirements.txt`. |
| 5 | (checkpoint) | Train ResNet18 from scratch instead of using `PyTorch_CIFAR10` checkpoint | The recommended checkpoint has a different forward graph (initial `maxpool`, single ReLU per block) and gives 21% accuracy in the marza96 ResNet18 before any compression. |

## How to reproduce

The full pipeline is in `notebooks/model_folding_pipeline.ipynb`. Open it in Google Colab with a T4 GPU and run cells in order. Total runtime: about 2.5 hours.

The notebook is split into steps:
1. Check GPU and mount Drive
2. Clone the upstream repo (pinned to commit 33e2818)
3. Install missing dependency (`thop`)
4. Apply compatibility patches (Patch 1, 2, 3 above)
5. Sanity-check the model definition
6. Train ResNet18 on CIFAR10 from scratch (~40 min)
7. Run the three REPAIR variants (~1.5 hours total)
8. Collect results and plot Figure 1 and Figure 2
9. Attempted VGG11-BN extension (training worked, sweep did not finish)

## Repository structure

```
.
├── README.md                              this file
├── report/
│   └── Final_report_v2.pdf                final report
├── notebooks/
│   └── model_folding_pipeline.ipynb       end-to-end Colab notebook
└── results/
    ├── baseline_results.csv               14 sparsity points x 3 variants
    ├── fig1_baseline_comparison.png       Figure 1
    └── fig2_ar_breakdown.png              Figure 2
```

## Citation

If you use this stress test, please also cite the original paper:

```bibtex
@inproceedings{wang2025forget,
  title     = {Forget the Data and Fine-tuning! Just Fold the Network to Compress},
  author    = {Wang, Dong and \v{S}iki\'{c}, Haris and Thiele, Lothar and Saukh, Olga},
  booktitle = {ICLR},
  year      = {2025}
}
```

## Use of AI assistance

I used Claude (Anthropic) for the following: debugging the four compatibility patches in the patches table above, polishing the English of the report, and discussing how to structure the analysis. All experimental design choices, model training, sweeps, result interpretation, and conclusions were performed by me.
