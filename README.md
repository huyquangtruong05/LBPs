<!-- # Project Setup

```bash
step 1 : Move the separately submitted datasets folder to the same level as the main_code.py file.
step 2 : python -m venv venv
step 3 : venv\Scripts\activate
step 4 : pip install -r requirements.txt
step 5 : python main_code.py
``` -->

# Symmetric LBP (SyLBP4) – Reimplementation & Experimental Analysis

This project is a full reimplementation and extended evaluation of the research paper:  
**The use of the symmetric finite difference in the local binary pattern (symmetric LBP)**  
(Zeinab Sedaghatjoo, Hossein Hosseinzadeh, 2024)

The goal of this project is to reproduce the core idea of the paper, verify its mathematical claims, and evaluate whether the proposed **4-bit Symmetric Local Binary Pattern (SyLBP4)** can replace the traditional **8-bit Standard LBP (StLBP)** while significantly reducing feature dimensionality and computational cost.

---

## Project Overview

Local Binary Pattern (LBP) is a classical texture descriptor widely used in:

- Face Detection
- Facial Expression Recognition
- Texture Classification
- Lightweight Computer Vision Systems

Traditional LBP compares each neighboring pixel with the center pixel, generating an **8-bit code (256 patterns)**.

The paper proposes a better alternative:

### Symmetric LBP (SyLBP)

Instead of comparing each pixel to the center, SyLBP compares **symmetric opposite pixel pairs**:

- Higher-order approximation of local gradients
- Better mathematical formulation
- Reduced redundancy
- Smaller feature vectors

The most important variant is:

### SyLBP4

Uses only **4 independent directional derivatives** instead of 8.  
This reduces:

- **256 patterns → 16 patterns**
- **4096 features → 256 features**
- Faster extraction
- Lower RAM usage
- Nearly identical classification accuracy

---

## Implemented Methods

This project includes 3 feature extraction methods:

| Method | Description                               | Output Patterns |
| ------ | ----------------------------------------- | --------------- |
| StLBP  | Standard Local Binary Pattern             | 256             |
| SyLBP8 | Symmetric LBP with 8 directions           | 256             |
| SyLBP4 | Optimized Symmetric LBP with 4 directions | 16              |

---

## Experimental Tasks

### 1. Face Detection

Binary classification:

- Face
- Non-face (clutter / background)

**Dataset sources:**

- CFD
- CFD-MR
- CFD-INDIA
- Custom clutter images

### 2. Facial Expression Recognition

5-class emotion recognition using CK+48:

- Angry
- Fear
- Happy
- Sadness
- Surprise

---

## Extended Analysis Beyond the Paper

This project does more than simply reproduce the paper.

### Added Experiments

#### Weakness Testing

Stress-testing SyLBP4 under:

- Salt & Pepper Noise
- Gaussian Blur
- Rotation Variance

#### Statistical Validation

- 5-Fold Cross Validation
- GridSearchCV
- Paired T-Test
- Pearson Correlation Analysis

#### Hybrid Deep Learning Bonus

Compare:

- CNN on raw pixels
- SyLBP4 + MLP

To test whether handcrafted descriptors still remain competitive.

---

## Key Findings

### Massive Compression

| Method | Feature Size |
| ------ | ------------ |
| StLBP  | 4096         |
| SyLBP4 | 256          |

➡ **16× smaller**

### Faster Extraction

SyLBP4 is significantly faster than Standard LBP due to fewer comparisons and smaller histograms.

### Accuracy Retention

Despite compression, SyLBP4 achieves nearly identical accuracy to StLBP on both tasks.

### Verified Redundancy Claim

The opposite symmetric bits are strongly negatively correlated:

- Pearson correlation ≈ **-0.95**

This validates the paper’s claim that half of the bits are redundant.

---

## Technologies Used

- Python 3.10+
- NumPy
- OpenCV
- SciPy
- Scikit-learn
- Matplotlib
- TensorFlow / Keras

---

## Project Structure

```bash
project/
├── main_code.py
├── requirements.txt
├── README.md
├── datasets/
│   ├── CK+48/
│   ├── CFD/
│   ├── clutter/
├── results/
```

## Installation & Setup

**Step 1: Download & Place Dataset Folder**  
Download `datasets.zip` from this [link_datasets](https://drive.google.com/file/d/1m7bEnq9cGxwJ1pAGOeTudQwfpsyX_PWj/view?usp=sharing).  
Extract the zip file and place the `datasets/` folder at the same directory level as `main_code.py`.

**Step 2: Create Virtual Environment**

```bash
python -m venv venv
```

**Step 3: Activate Environment (Windows)**

```bash
venv\Scripts\activate
```

**Step 4: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 5: Run Project**

```bash
python main_code.py
```

## Expected Outputs

After execution, the program may generate:

- Accuracy reports
- Confusion matrices
- Comparison charts
- Weakness analysis plots
- Statistical test results
- Hybrid model evaluation
- Saved figures in `results/`

---

## Why This Project Matters

This project demonstrates strong understanding in:

**Computer Vision**

- Feature engineering
- Texture descriptors
- Face recognition pipelines

**Machine Learning**

- SVM tuning
- Cross-validation
- Statistical significance testing

**Research Reproduction**

- Re-implementing academic papers
- Verifying claims experimentally
- Critical analysis beyond original work

**Engineering Mindset**

- Optimization
- Efficiency vs Accuracy trade-off
- Real-world robustness testing

---

## Real-World Use Cases

SyLBP4 is suitable for:

- Edge AI devices
- Embedded cameras
- Low-memory systems
- Real-time face detection
- Lightweight recognition systems

---

## Future Improvements

Possible next steps:

- Rotation-invariant SyLBP
- Multi-scale SyLBP
- MB-LBP integration
- Mobile deployment
- Real-time webcam system
- Larger benchmark datasets

---

## Authors

- **Truong Quang Huy**
- **Nguyen Bach Khoa**

**University of Science – VNUHCM (HCMUS)**
