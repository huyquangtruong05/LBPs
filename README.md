# Symmetric LBP (SyLBP4) – Reimplementation, Benchmarking & CNN Comparison

## Overview

This notebook-based project reimplements and extends the 2024 paper **The use of the symmetric finite difference in the local binary pattern (symmetric LBP)**.

The core objective is to verify whether **SyLBP4** (4-bit symmetric local binary pattern) can preserve recognition performance while drastically reducing feature size and extraction cost compared with traditional **Standard LBP (StLBP)**.

In addition to classical machine learning experiments, this project also includes a bonus deep-learning comparison:

- Raw image CNN
  n- SyLBP4 feature map + CNN/MLP pipeline

---

## Implemented Descriptors

| Method | Idea                                    | Output Patterns |
| ------ | --------------------------------------- | --------------- |
| StLBP  | Compare neighbors with center pixel     | 256             |
| SyLBP8 | Compare opposite symmetric pairs        | 256             |
| SyLBP4 | Keep 4 independent symmetric directions | 16              |

### Why SyLBP4 Matters

- 16× smaller histogram representation (4096 → 256 in block histograms)
- Fewer comparisons
- Lower memory usage
- Faster training / inference
- Competitive accuracy

---

## Datasets Used

## 1. CK+48 Emotion Recognition

5 classes:

- Angry
- Fear
- Happy
- Sadness
- Surprise

## 2. CFD Face Detection

Binary classification:

- Face
- Non-face / Clutter

Expected folder names:

```bash
datasets/
├── CK_Plus/
├── CFD/
└── Clutter_Images/
```

---

## Main Experimental Pipeline

The notebook executes the following stages:

### A. Data Loading

Custom loaders prepare images and labels from dataset folders.

### B. Preprocessing

- Resize / normalize
- Grayscale processing
- Feature-ready image formatting

### C. Feature Extraction

Implemented manually in Python:

- Standard LBP
- Symmetric LBP8
- Symmetric LBP4

### D. Classical ML Evaluation

Using SVM classifiers with cross-validation and metrics such as:

- Accuracy
- Classification report
- Confusion matrix
- Extraction time

### E. Redundancy Validation

Measures correlation between opposite symmetric bits to verify the paper's redundancy claim.

### F. Robustness / Weakness Analysis

Stress tests SyLBP4 under:

- Salt & Pepper noise
- Gaussian blur
- Rotation changes

### G. Bonus Deep Learning Comparison

Two pipelines are compared:

1. Raw CNN on images
2. SyLBP4 spatial representation + neural model

### H. Visualization Dashboard

The notebook exports charts such as:

- Accuracy bar charts
- Heatmaps
- Runtime comparison
- CNN comparison plots

---

## Technologies

- Python 3.10+
- NumPy
- OpenCV
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- TensorFlow / Keras
- Pandas

---

## Project Structure

```bash
project/
├── main.ipynb
├── README.md
├── requirements.txt
├── datasets/
└── results/
```

---

## Setup & Run

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate environment (Windows)

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open `main.ipynb` and run all cells.

---

## Expected Outputs

Inside `results/`:

- JSON result summaries
- Accuracy plots
- Heatmaps
- Runtime charts
- CNN comparison figures
- Weakness analysis charts

---

## Key Contributions

- Reproduced an academic computer vision paper
- Verified theoretical redundancy claims experimentally
- Benchmarked efficiency vs accuracy trade-offs
- Added robustness testing beyond the original paper
- Added handcrafted vs deep learning comparison

---

## Future Work

- Rotation-invariant SyLBP
- Multi-scale SyLBP
- Real-time webcam deployment
- Mobile / edge optimization
- Larger benchmark datasets

---

## Authors

- Truong Quang Huy
- Nguyen Bach Khoa

University of Science – VNUHCM (HCMUS)
