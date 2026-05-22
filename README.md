# PCA + ANN Face Recognition System

Implementation of the **Turk & Pentland (1991) Eigenfaces** method combined  
with a custom **Backpropagation Neural Network** for face classification.

---

## Project Overview

| Component | Details |
|-----------|---------|
| Feature extraction | PCA via surrogate covariance (Eigenfaces) |
| Classifier | Multi-layer ANN with backpropagation |
| Libraries | NumPy · SciPy · OpenCV · Matplotlib |
| Train / Test split | 60 % / 40 % (stratified per subject) |

---

## Dataset

Download from:

```
https://github.com/robaita/introduction_to_machine_learning/blob/main/dataset.zip
```

Extract so the folder structure is:

```
dataset/
  s1/   1.pgm  2.pgm  ...  10.pgm
  s2/   1.pgm  ...
  ...
  s40/  ...
```

This is the standard **ORL / AT&T** database: 40 subjects × 10 images = 400 total.  
Each image is 92 × 112 pixels, grayscale.

---

## Files

```
face_recognition_pca_ann.py   ← Main implementation
smoke_test.py                 ← Quick sanity-check (synthetic data, no real dataset needed)
README.md                     ← This file
```

---

## Installation

```bash
pip install numpy scipy opencv-python matplotlib
```

---

## Quick Start

```bash
# 1. Verify the code works (no dataset needed)
python smoke_test.py

# 2. Run on the real dataset
python face_recognition_pca_ann.py --dataset ./dataset

# 3. Custom k values and more epochs
python face_recognition_pca_ann.py \
    --dataset ./dataset \
    --k_values 5 10 20 30 50 75 100 \
    --ann_epochs 1000 \
    --ann_lr 0.005 \
    --n_imposters 8 \
    --output_dir ./results
```

---

## Algorithm — Step by Step

### Training

| Step | Operation | Dimensions |
|------|-----------|-----------|
| 1 | Build face database `FaceDb` | `mn × p` |
| 2 | Compute mean face `M = mean(FaceDb, axis=1)` | `mn × 1` |
| 3 | Mean-zero: `A = FaceDb − M` | `mn × p` |
| 4 | Surrogate covariance: `C = AᵀA` *(Turk & Pentland trick)* | `p × p` |
| 5 | Eigen-decomposition of `C` → sort descending | `p × p` |
| 6 | Select top-`k` eigenvectors `Ψ` | `p × k` |
| 7 | Eigenfaces: `Φ = Ψᵀ · Aᵀ` | `k × mn` |
| 8 | Signatures: `ω = Φ · A` | `k × p` |
| 9 | Train ANN on `(ωᵀ, labels)` | `p × k` input |

### Testing (single image `I`)

```
I₁  = flatten(I)           # mn × 1
I₂  = I₁ − M              # mean-zero
Ω   = Φ · I₂              # project to eigenspace  (k × 1)
ŷ   = ANN.predict(Ωᵀ)     # predicted subject label
```

---

## Experiments

### (a) Accuracy vs k

The script trains a separate model for each `k` in `--k_values` and  
plots **recognition accuracy (%)** against the number of eigenfaces.

Output: `output/accuracy_vs_k.png`

Expected trend: accuracy rises with `k`, then plateaus or slightly  
drops (over-fitting / noise dimensions).

### (b) Imposter Detection

A subset of subjects (`--n_imposters`) are **not enrolled** during training.  
Their images are added to the test set as imposters.

Detection strategy — **reconstruction error**:

```
error(I) = ‖I₂ − Φᵀ·(Φ·I₂)‖₂
```

A face that cannot be reconstructed well from the eigenspace is far  
from the training distribution → likely an imposter.

An optimal threshold is found by maximising **balanced accuracy**  
(average of true-positive rate and true-negative rate).

Output: `output/imposter_detection.png`  
  - Histogram of genuine vs imposter reconstruction errors  
  - ROC curve (Imposter Rejection Rate vs False Acceptance Rate)

---

## ANN Architecture

```
Input (k)  →  Hidden-1 (sigmoid)  →  Hidden-2 (sigmoid)  →  Output (softmax)
```

- **Hidden sizes**: auto-computed as `[4·n_classes, 2·n_classes]`
- **Loss**: categorical cross-entropy
- **Optimiser**: mini-batch SGD (batch = 32)
- **Weight init**: Xavier

---

## Output Files

| File | Description |
|------|-------------|
| `eigenfaces.png` | Grid of the top eigenface components + mean face |
| `accuracy_vs_k.png` | Accuracy (%) vs number of eigenfaces k |
| `imposter_detection.png` | Error histogram + ROC curve for imposter detection |

---

## References

1. M. Turk and A. Pentland, *"Eigenfaces for Recognition"*, Journal of  
   Cognitive Neuroscience, 3(1):71–86, 1991.

2. ORL / AT&T Face Database,  
   https://cam-orl.co.uk/facedatabase.html
