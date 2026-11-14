# ☀️ Solar Filament Segmentation — Kaggle Competition 2026

### Instance segmentation of solar filaments in full-disk GONG H-alpha observations

<p align="center">

**U-Net + Pretrained ResNet-50 + Multi-Annotator Training + Morphological Instance Extraction**

</p>

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/filament-segmentation-2026)
[![Computer Vision](https://img.shields.io/badge/Computer-Vision-orange)]()
[![Segmentation](https://img.shields.io/badge/Task-Instance%20Segmentation-green)]()

</p>

---

## 🏆 Final Result

| Metric | Result |
|---|---:|
| 🥇 Public Kaggle PQ | **0.34** |
| 🔬 Best date-grouped validation PQ | **0.4435** |
| 🧠 Final architecture | **U-Net + ResNet-50** |
| 🖼️ Training resolution | **1024 × 1024** |
| 🌞 Original resolution | **2048 × 2048** |
| 📚 Training images | **707** |
| 🧑‍🔬 Annotation records | **1,154** |
| 🧪 Test images | **180** |

The final model was trained on **all 707 labelled images** and submitted to the public Kaggle leaderboard.

> **Official public leaderboard result: 0.34 PQ**

The complete inference pipeline, trained weights, notebooks, results and technical report are provided in this repository.

---

# 🔬 Project Overview

The goal of this project is to automatically identify and segment individual **solar filaments** in full-disk H-alpha observations from the Global Oscillation Network Group (GONG).

Solar filaments are elongated structures appearing as dark features against the solar disk. Their segmentation is challenging because they can be:

- thin and elongated,
- faint and low contrast,
- fragmented,
- close to or touching other filaments,
- inconsistently annotated by different human annotators.

This project investigates a reproducible deep-learning pipeline for the challenge using a **U-Net architecture with a pretrained ResNet-50 encoder**, multi-annotator training, test-time augmentation, and morphology-based instance extraction.

---

# 👁️ Qualitative Results

The following examples show the input H-alpha images, annotation records and predicted filament instances.

<p align="center">
  <img src="results/fig_examples.png" alt="Solar filament segmentation examples" width="100%">
</p>

### What the visualization shows

The model produces a pixel-level filament probability map and then converts this semantic prediction into individual filament instances.

Different colours represent different predicted instances.

The examples demonstrate that the model can recover:

- curved filament structures,
- multiple filaments within one solar disk,
- faint filament regions,
- elongated structures,
- spatially separated instances.

The remaining errors are mainly associated with ambiguous boundaries, small filaments, fragmented structures and cases close to the official IoU = 0.5 matching threshold.

---

# 📈 Training Behaviour

The training process was monitored using loss and validation metrics throughout the experiments.

<p align="center">
  <img src="results/learning_curves.png" alt="Training and validation learning curves" width="95%">
</p>

The learning curves were also useful for diagnosing an instability in the initial U-Net experiments.

Some early runs exhibited validation-loss spikes and an epoch with PQ = 0.000. Investigation showed that BatchNorm running statistics were unstable. Re-estimating the BatchNorm statistics restored the affected checkpoint to approximately PQ = 0.27 without changing the learned weights.

This stabilisation was incorporated into the final pipeline.

---

# 🧠 Final Architecture

The final model is a **U-Net with a pretrained ResNet-50 encoder**.

```text
                         ┌──────────────────────┐
                         │   2048 × 2048 Image  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Resize → 1024 × 1024 │
                         │ 1–99% Normalisation  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                  ┌─────────────────────────────────┐
                  │       U-Net + ResNet-50         │
                  │       ImageNet pretrained       │
                  └───────────────┬─────────────────┘
                                  │
                                  ▼
                         Probability Map
                                  │
                     ┌────────────┼────────────┐
                     │            │            │
                  Original       H-Flip       V-Flip
                     │            │            │
                     └────────────┼────────────┘
                                  │
                                  ▼
                       Average Predictions
                                  │
                                  ▼
                        Upsample → 2048²
                                  │
                                  ▼
                        Threshold = 0.5
                                  │
                                  ▼
                  Morphological Closing (r = 10)
                                  │
                                  ▼
                       8-Connected Components
                                  │
                                  ▼
                    Remove Objects ≤ 400 px
                                  │
                                  ▼
                         COCO Compressed RLE
                                  │
                                  ▼
                         submission.csv
```

---

# 🧬 Multi-Annotator Training

A particularly important aspect of this competition is that the same image may contain multiple annotation records from different annotators.

Instead of fixing one annotation record permanently, the final training pipeline randomly selects an annotator record for each image during every epoch.

The selected polygons are converted into a union segmentation target.

```text
              One solar image
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      Annotator A Annotator B Annotator C
          │         │         │
          └─────────┼─────────┘
                    ▼
          Random record per epoch
                    │
                    ▼
              Union mask
                    │
                    ▼
                U-Net
```

This design follows the structure of the official evaluation, which evaluates predictions against annotator records separately.

---

# 📊 Experimental Progress

The project was developed through a sequence of controlled experiments.

| Experiment | Configuration | Validation PQ | Public PQ |
|---|---|---:|---:|
| Baseline | U-Net from scratch | ~0.290 | **0.21** |
| Stabilisation | BN + TTA | ~0.304 | **0.22** |
| Small-object filtering | min 200 px | ~0.351 | **0.26** |
| Morphological recipe | min 400 px + closing | **0.372** | **0.28** |
| Multi-annotator training | Random record + BN | **0.384** | **0.29** |
| B03 | ResNet-34 U-Net | **0.433** | **0.33** |
| B04 | ResNet-50 U-Net | **0.433–0.444** | **0.33** |
| Ensemble | ResNet-34 + ResNet-50 | ~0.439 | **0.33** |
| **F01 — Final** | **ResNet-50 + all 707 images** | — | **🏆 0.34** |

### Key findings

The experiments showed that:

**1. Post-processing was highly important**

Validation PQ increased from approximately:

`0.29 → 0.37`

after introducing the selected morphological instance-extraction pipeline.

**2. Pretrained encoders provided another major improvement**

The transition from the scratch U-Net to pretrained ResNet encoders increased validation PQ from approximately:

`0.38 → 0.43`

and the public leaderboard from:

`0.29 → 0.33`.

**3. Ensembling was not reliably beneficial**

The B03/B04 ensemble produced approximately 0.439 compared with 0.4435 for the strongest individual model.

**4. Confidence filtering was not consistently useful**

A small improvement appeared on one validation half, but it did not survive the split-half check.

**5. Retraining on all available labelled data produced the final 0.34 public score.**

---

# 🎯 Official Evaluation Metric

The competition uses a custom **many-to-many Panoptic Quality (PQ)** metric.

A prediction and ground-truth instance form a true-positive pair when:

```text
IoU > 0.5
```

Importantly, the matching is **many-to-many**, rather than one-to-one.

The metric is:

$$
PQ =
\frac{\sum IoU_{TP}}
{TP + 0.5FP + 0.5FN}
$$

Predictions are evaluated separately against every annotator record associated with an image, and the resulting quantities are pooled.

The implementation used in this project was independently cross-checked against the organizers' scoring code and reproduced the official validation score to six decimal places.

---

# 🔍 Error Analysis

The strongest held-out ResNet-50 configuration produced:

| Quantity | Result |
|---|---:|
| TP | **1,063** |
| FP | **577** |
| FN | **544** |
| Mean matched IoU | **0.677** |
| Detection Quality | **0.655** |
| PQ | **0.4435** |

The remaining errors were not simply caused by poor pixel segmentation.

A large proportion of false positives were predictions that overlapped an annotation but failed the strict IoU > 0.5 criterion.

Similarly, many missed filaments had a prediction nearby but with insufficient IoU.

### Small filaments

Objects between approximately **200–999 pixels** were particularly difficult, producing high false-positive and false-negative rates.

### Fragmentation

Only a relatively small subset of missed objects could be fixed simply by merging fragments, suggesting that fragmentation alone is not the dominant source of error.

### Annotation disagreement

On multiply annotated validation images, one annotator scored against another under the same metric produced approximately:

**PQ ≈ 0.38**

This provides evidence that annotation ambiguity is an important characteristic of the task.

It is treated as an agreement reference rather than a theoretical upper bound on model performance.

---

# 🔄 Validation vs. Public Leaderboard

One interesting observation throughout the experiments was a consistent gap between validation and public leaderboard performance.

```text
Validation PQ
      │
      │       ███████████████████
      │       ~0.43–0.44
      │
      │
      │
      │       ~0.33–0.34
      │       ██████████████
      │
      └──────────────────────────
            Public Leaderboard
```

The absolute validation score was approximately 0.09–0.10 higher than the public leaderboard.

However, the relative improvement between experiments generally transferred in the same direction.

The final model achieved:

**Public Kaggle PQ = 0.34**

---

# 🧪 Reproducibility

This repository is designed to reproduce the submitted predictions from the **stored final model weights** without retraining.

The final weights are stored in half precision and are approximately **65 MB**.

The reproduction pipeline was independently checked against the submitted predictions and achieved approximately:

**0.9994 pooled PQ agreement**

between the reproduced and submitted files.

This provides a verification that the published inference pipeline reproduces the final submission closely.

---

# 🚀 Reproducing the Submission

## 1. Clone the repository

```bash
git clone https://github.com/parsaBehjatTabrizi/Solar-Filament-Segmentation-Kaggle-Competition2026.git

cd Solar-Filament-Segmentation-Kaggle-Competition2026
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Obtain the competition data

Download the official competition dataset through Kaggle after accepting the competition rules.

The competition dataset itself is not redistributed in this repository.

## 4. Open the reproduction notebook

Run:

```text
reproduce_submission.ipynb
```

The notebook loads the stored final weights and performs inference on the 180 test images.

### No retraining is required.

The pipeline generates:

```text
submission.csv
```

using the official COCO compressed RLE format.

---

# 📁 Repository Structure

```text
Solar-Filament-Segmentation-Kaggle-Competition2026/
│
├── 📓 notebooks/
│   ├── dataset analysis
│   ├── metric / evaluation analysis
│   ├── model experiments
│   └── reproduce_submission.ipynb
│
├── 🧠 model / source code
│   ├── dataset
│   ├── preprocessing
│   ├── model
│   ├── training
│   ├── evaluation
│   ├── post-processing
│   └── RLE generation
│
├── 💾 weights/
│   └── final ResNet-50 U-Net weights
│
├── 📊 results/
│   ├── fig_examples
│   ├── learning_curves
│   ├── validation results
│   └── experiment results
│
├── 📄 report/
│   └── technical report
│
├── 📤 submission/
│   └── submission.csv
│
└── 📦 requirements.txt
```

---

# 🖼️ Example Prediction

The following visualization shows the complete qualitative comparison:

```text
┌─────────────────┬────────────────────┬─────────────────────┐
│ Input H-alpha   │ Annotator Record    │ Model Prediction    │
├─────────────────┼────────────────────┼─────────────────────┤
│                 │                    │                     │
│   ☀️ Solar      │   Human-labelled   │   🧠 Predicted      │
│   observation   │   filament masks   │   instances          │
│                 │                    │                     │
└─────────────────┴────────────────────┴─────────────────────┘
```

For the complete set of examples, see:

**`results/fig_examples.png`**

---

# 📈 Learning Curves

Training behaviour and validation metrics are visualised in:

**`results/learning_curves.png`**

These plots document the training dynamics and were also used during development to identify the BatchNorm instability observed in early experiments.

---

# ⚠️ Limitations

The final approach has several important limitations.

### Semantic segmentation followed by connected components

The network predicts a semantic filament probability map rather than explicitly learning individual object identities.

Consequently, touching filaments can be difficult to separate.

### Small filaments

Thin and small structures remain one of the major sources of errors.

### Annotation uncertainty

Different annotators can produce substantially different filament boundaries and object selections.

### Validation split

Hyperparameters were selected using a date-grouped validation split. Although this reduces temporal leakage, it is still a single development split.

---

# 🔮 Future Work

Several directions could potentially improve the solution:

- 🔹 Instance-aware segmentation architectures
- 🔹 Boundary-aware segmentation heads
- 🔹 Distance-transform or skeleton-based auxiliary losses
- 🔹 Probabilistic multi-annotator targets
- 🔹 Higher-resolution or tiled inference
- 🔹 Cross-validation over observation-date blocks
- 🔹 Better modelling of small and faint filaments
- 🔹 Explicit separation of touching filament structures

---

# 🛠️ Technologies

- **Python**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **OpenCV / image processing**
- **Pycocotools**
- **Matplotlib**
- **Kaggle GPU infrastructure**
- **U-Net**
- **ResNet-50**
- **ImageNet pretrained weights**

---

# 📚 References

1. Ahmadzadeh et al. *A dataset of manually annotated filaments from H-alpha observations*. Scientific Data, 2024.
2. Ahmadzadeh et al. *Solar Filament Segmentation Challenge 2026*. Kaggle.
3. Ronneberger et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI, 2015.
4. He et al. *Deep Residual Learning for Image Recognition*. CVPR, 2016.
5. Ioffe & Szegedy. *Batch Normalization*. ICML, 2015.
6. Kingma & Ba. *Adam: A Method for Stochastic Optimization*. ICLR, 2015.
7. Kirillov et al. *Panoptic Segmentation*. CVPR, 2019.
8. Milletari et al. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*. 3DV, 2016.

---

# 👤 Author

### Parsa Behjat Tabrizi

Computer Science / Computer Engineering

Islamic Azad University, Pardis Branch

📧 **behjattabrizi.sp@gmail.com**

---

<p align="center">

### ☀️ Solar Filament Segmentation Challenge 2026

**Deep Learning · Computer Vision · Instance Segmentation · Solar Physics**

</p>

<p align="center">

⭐ If you find this project useful, consider starring the repository.

</p>
