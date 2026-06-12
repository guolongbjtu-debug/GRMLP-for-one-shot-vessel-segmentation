# GRMLP (Gated Residual MLP): Curvature-Guided Deep Learning Framework for one-shot 2D Vessel Segmentation

A dual-branch MLP that fuses **multi-scale Hessian features** (coarse vessels) with **local neighborhood features** (fine vessels and endings) via an attention-gated residual fusion block, producing per-pixel segmentation probabilities through a Sigmoid output.

---

## Directory Structure

```
CPODL_git/
├── scripts/                       # Pipeline entry points
│   ├── get_config.m               # Global config: dataset, params, paths
│   ├── prepare_dataset.m          # Dataset preprocessing / normalization
│   ├── GRMLP_multiscale_training.m
│   └── GRMLP_multiscale_inference.m
├── GRMLP_Architecture/
│   └── GRMLP.m                    # Network definition and training logic
├── units/                         # Utilities
│   ├── Filter_*.m                 # Vessel filters (Frangi / Jerman / Zhang / Guo)
│   ├── hessian_*.m                # Hessian matrix / eigenvalues
│   ├── preprocessing_*.m          # Binarize, grayscale, remove small regions
│   ├── postprocessing_*.m         # Connected-component filter, erode, dilate, fill holes, deburr
│   ├── calculate_*.m              # IoU / Dice / Recall-Dice curves
│   └── extract_neighborhood.m     # Neighborhood feature extraction
├── dataset/                       # Example dataset (DRIVE / DRIVE_enhenced)
├── dataset_normalization/         # Normalized dataset
├── checkpoints/                   # Trained networks + statistics
└── result/                        # Inference output (probability/binary maps, Dice)
```

---

## Workflow

**(1) Add the dataset** under `dataset/<DATASET>/`:

```
dataset/<DATASET>/
├── training/
│   ├── image/      # original
│   ├── mask/       # RoI mask
│   └── target/     # vessel annotation
└── test/
    ├── image/
    ├── mask/
    └── target/
```

- Formats: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.gif`
- One-shot setup: include only **one** training image.

**(2) Configure** — edit [scripts/get_config.m](scripts/get_config.m) (dataset name, `maxsigma`, etc.), then click **Run** ▶ to set up paths automatically. 

**(3) Preprocess** — open `prepare_dataset.m` and click **Run** ▶. It writes standardized data to `dataset_normalization/<DATASET>/` — 512×512 grayscale images in [0,1], binary masks/targets.

**(4) Train** — open `GRMLP_multiscale_training.m` and click **Run** ▶. The trained network and parameters are saved to `checkpoints/`.

**(5) Infer** — set `cfg.testindex` and `cfg.testtime` in `get_config.m`, then open `GRMLP_multiscale_inference.m` and click **Run** ▶. It loads the matching checkpoint and writes the results (probability/binary maps) to `result/`. The current version only supports inference on one image per run.

---

## Requirements

- MATLAB 2024b with Deep Learning Toolbox
- Training: `trainNetwork` + Adam optimizer · Inference: `predict`
