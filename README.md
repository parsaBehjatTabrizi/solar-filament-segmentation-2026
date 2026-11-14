# Solar Filament Segmentation — Kaggle Competition 2026

**A U-Net with a pretrained ResNet-50 encoder, multi-annotator training, and morphology-based instance extraction**

This repository contains my solution to the **Solar Filament Segmentation Challenge 2026**, a Kaggle competition focused on automatic segmentation of solar filaments in full-disk GONG H-alpha observations.

The task is challenging because the dataset contains multiple annotation records for the same observation, while the official evaluation scores predictions independently against each annotator record using a custom many-to-many Panoptic Quality (PQ) metric.

## Final Result

| Evaluation | PQ |
|---|---:|
| Best date-grouped validation | **0.4435** |
| Final public Kaggle leaderboard | **0.34** |

The final submitted model is a **U-Net with a pretrained ResNet-50 encoder**, trained on all 707 labelled images.

The final inference pipeline uses:

**ResNet-50 U-Net → flip TTA → probability averaging → thresholding → morphological closing → connected components → small-object filtering → COCO RLE**

The trained weights and an inference-only notebook are included so that the submitted predictions can be reproduced **without retraining the model**.

---

## 1. Competition

The Solar Filament Segmentation Challenge 2026 is an instance-segmentation problem using full-disk **GONG H-alpha** solar observations.

The dataset contains:

- **707 training images**
- **180 test images**
- **1,154 annotation records**
- Original image resolution: **2048 × 2048**
- Multiple annotator records for some observations

Each annotation record represents the polygons drawn by one annotator for one image.

The final submission contains one predicted mask per detected filament using COCO compressed RLE encoding.

---

## 2. Main Challenges

There are two important difficulties in this problem.

### Multi-annotator annotations

The same image can have multiple annotation records. Different annotators may disagree about:

- whether a filament exists,
- its exact boundaries,
- whether nearby structures form one filament or several,
- and the extent of faint or small filaments.

Because the official metric evaluates predictions separately against each annotator record, the training strategy needs to account for this variability.

### Custom Panoptic Quality metric

The organizers use a custom pooled Panoptic Quality metric with **many-to-many matching**.

For every ground-truth/prediction pair:

- IoU must be **strictly greater than 0.5** to count as a true positive.
- Every qualifying pair contributes its IoU to the numerator.
- Predictions without a qualifying match are false positives.
- Ground-truth instances without a qualifying prediction are false negatives.
- Results are pooled across annotator records.

The implementation used in this project was independently cross-checked against the organizers' evaluation code and reproduced the official score to six decimal places on validation predictions.

---

# 3. Method

## 3.1 Model

The final model is a **U-Net with a pretrained ResNet-50 encoder**.

The encoder is initialized with public ImageNet weights.

The one-channel grayscale H-alpha image is repeated to three channels and ImageNet-normalised inside the network.

The network contains:

- ResNet-50 encoder
- U-Net decoder
- skip connections
- bilinear upsampling
- double-convolution decoder blocks
- final 1×1 convolution
- per-pixel filament probability output

The model contains approximately **32.6 million parameters**.

---

## 3.2 Training Data Strategy

Instead of always training against one fixed annotator, a training image with multiple annotation records randomly selects **one annotator record per epoch**.

The polygons from that record are converted into a union segmentation mask.

This strategy was chosen because the official metric evaluates each annotator record independently.

The development dataset was divided using an observation-date-grouped split:

- **568 training images**
- **139 validation images**
- **225 validation annotator records**
- random seed: **42**

Grouping by observation date prevents closely related observations from appearing in both training and validation.

The 180 test images were never used for model or hyperparameter decisions.

For the final submission, the selected configuration was retrained using **all 707 labelled images**.

---

## 3.3 Preprocessing

The original 2048 × 2048 images are resized to:

**1024 × 1024**

using area interpolation.

Each image is independently normalised using its:

- 1st intensity percentile
- 99th intensity percentile

The only training augmentation used in the final pipeline is:

- horizontal flipping
- vertical flipping

---

## 3.4 Loss and Optimisation

The model is trained using:

**Binary Cross-Entropy + Soft Dice Loss**

Optimisation:

- Optimizer: Adam
- Encoder learning rate: `1e-4`
- Decoder learning rate: `1e-3`
- Weight decay: `1e-5`
- Learning-rate schedule: cosine decay
- Training duration: 50 epochs
- Batch size: 4
- Mixed precision training

---

## 3.5 Batch-Normalisation Stabilisation

Early experiments revealed unstable validation behaviour.

Some epochs produced validation-loss spikes, including a checkpoint with PQ = 0.000.

Investigation showed that the learned model weights themselves were not necessarily the problem; the BatchNorm running statistics were unstable.

Re-estimating the BatchNorm statistics using approximately 240 training images restored the affected checkpoint from PQ = 0.000 to approximately **0.27**.

Therefore, BatchNorm statistics were re-estimated before validation and for the final model while keeping the learned weights unchanged.

---

# 4. Inference Pipeline

The final inference process is:

```text
2048 × 2048 H-alpha image
            │
            ▼
      Resize to 1024²
            │
            ▼
      1–99% normalisation
            │
            ▼
     ResNet-50 U-Net
            │
       ┌────┼────┐
       │    │    │
    original  H-flip
              V-flip
             HV-flip
       └────┼────┘
            ▼
     Average probabilities
            │
            ▼
       Upsample to 2048²
            │
            ▼
      Threshold at 0.5
            │
            ▼
 Morphological closing (r = 10)
            │
            ▼
  8-connected components
            │
            ▼
 Remove components ≤ 400 pixels
            │
            ▼
       COCO compressed RLE
            │
            ▼
       submission.csv
```

The four probability maps from the original image and its three flip transformations are averaged before instance extraction.

Morphological closing is used to reconnect fragments belonging to the same filament.

Small connected components of **400 pixels or fewer** are discarded.

---

# 5. Experimental Development

The final solution was developed through a sequence of controlled experiments.

| Configuration | Validation PQ | Public PQ |
|---|---:|---:|
| Plain U-Net from scratch | ~0.290 | 0.21 |
| BN stabilisation + TTA | ~0.304 | 0.22 |
| Remove objects < 200 px | ~0.351 | 0.26 |
| 400 px + morphological closing | **0.372** | **0.28** |
| Random annotator training + BN fix | **0.384** | **0.29** |
| ResNet-34 U-Net | **0.433** | **0.33** |
| ResNet-50 U-Net | **0.433–0.444** | **0.33** |
| ResNet-34 + ResNet-50 ensemble | ~0.439 | 0.33 |
| ResNet-50 trained on all 707 images | — | **0.34** |

The final model was selected based on the controlled validation experiments and then retrained on all available labelled images.

---

# 6. Post-processing Experiments

Post-processing produced one of the largest improvements in the project.

Starting from the raw U-Net baseline:

**Validation PQ: ~0.29**

After the selected post-processing recipe:

**Validation PQ: ~0.37**

The parameters were selected using the validation data with a split-half check over observation dates.

The tested ranges included:

- probability threshold: 0.4–0.6
- minimum component size: 200–800 pixels
- closing radius: 0–15 pixels

The final configuration was:

```text
Probability threshold = 0.5
Minimum component size = 400 px
Morphological closing radius = 10 px
```

---

# 7. Error Analysis

Detailed error analysis showed that the remaining errors are not simply caused by poor pixel-level segmentation.

For the strongest held-out ResNet-50 configuration:

- TP = **1,063**
- FP = **577**
- FN = **544**
- Mean matched IoU = **0.677**
- Detection quality = **0.655**
- PQ = **0.4435**

Many errors are borderline cases close to the IoU = 0.5 matching threshold.

Among the false positives:

- 365 overlapped an annotated filament but had IoU ≤ 0.5
- 212 had no overlap with an annotation

Among missed ground-truth instances:

- 220 overlapped exactly one prediction but had IoU ≤ 0.5
- 243 had no overlapping prediction

Small filaments were particularly difficult:

- objects between 200–999 pixels had high false-positive and false-negative rates.

The analysis also showed that only a relatively small fraction of missed objects could be corrected simply by merging fragmented predictions.

---

# 8. Annotator Agreement

The dataset contains genuine disagreement between human annotators.

On validation images with multiple annotation records, comparing one annotator against another under the same official metric produced approximately:

**PQ = 0.38**

This is treated as an **agreement reference**, not as a theoretical upper bound for machine performance.

The result demonstrates that some of the remaining model errors are associated with ambiguity in the annotations themselves.

---

# 9. Validation vs. Kaggle Leaderboard

The validation and public leaderboard scores consistently showed a gap of approximately **0.09–0.10 PQ**.

However, the relative effect of the experiments was generally consistent between validation and the public leaderboard.

For example:

```text
Post-processing:
Validation  ~0.29 → ~0.37
Leaderboard ~0.21 → ~0.28

Pretrained encoder:
Validation  ~0.38 → ~0.43
Leaderboard ~0.29 → ~0.33
```

The final all-data model achieved a public leaderboard score of:

**PQ = 0.34**

The public leaderboard is based on a hidden subset of the test data, so the validation score should not be interpreted as an exact prediction of the final evaluation.

---

# 10. Reproducibility

The repository contains:

- source code
- training/inference configuration
- final model weights
- experiment notebooks
- submission generation code
- reproducibility notebook
- technical report

The final weights are stored in half precision and are approximately **65 MB**.

The inference notebook loads the stored weights and regenerates the submission without retraining.

The reproduced submission was compared with the submitted predictions and achieved a pooled PQ agreement of approximately:

**0.9994**

This provides a verification that the published inference pipeline reproduces the submitted predictions closely.

---

# 11. Repository Structure

```text
Solar-Filament-Segmentation-Kaggle-Competition2026/
│
├── notebooks/
│   ├── dataset analysis
│   ├── metric/evaluation analysis
│   ├── model experiments
│   └── reproduce_submission.ipynb
│
├── source code
│   ├── dataset / preprocessing
│   ├── model
│   ├── training
│   ├── evaluation
│   ├── post-processing
│   └── RLE generation
│
├── weights/
│   └── final ResNet-50 U-Net weights
│
├── results/
│   ├── validation results
│   ├── experiment results
│   └── prediction visualisations
│
├── submission/
│   └── submission.csv
│
├── report/
│   └── technical report
│
└── requirements.txt
```

---

# 12. Reproducing the Final Submission

The final pipeline is designed for **inference from stored weights** rather than retraining.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain the competition data

The competition dataset must be obtained through Kaggle after accepting the competition rules.

The original competition images and annotations are **not redistributed in this repository**.

### 3. Load the final weights

The reproduction notebook loads the stored ResNet-50 U-Net weights.

### 4. Run inference

Open:

```text
notebooks/reproduce_submission.ipynb
```

and run the inference pipeline.

The notebook:

1. loads the test images,
2. preprocesses them,
3. loads the trained model,
4. performs flip TTA,
5. averages the probability maps,
6. performs instance extraction,
7. applies the final post-processing,
8. encodes masks using COCO compressed RLE,
9. writes `submission.csv`.

No model retraining is required.

---

# 13. Limitations

The final approach is intentionally simple and reproducible, but it has several limitations.

### Semantic rather than explicit instance segmentation

The U-Net predicts a semantic filament probability map. Individual filaments are subsequently extracted using connected components.

Therefore, the model does not explicitly learn object boundaries or instance identities.

This makes touching filaments difficult to separate.

### Small and faint filaments

Small structures remain a major source of false positives and missed annotations.

Higher-resolution processing or tiled inference could potentially improve representation of these objects.

### Multi-annotator uncertainty

The model is trained using randomly selected annotator records rather than an explicit probabilistic model of annotator uncertainty.

Future work could investigate:

- probabilistic multi-annotator targets,
- boundary-aware segmentation,
- instance-aware architectures,
- higher-resolution/tiled inference,
- cross-validation over observation-date blocks.

---

# 14. Future Work

The experiments suggest several promising directions:

1. **Instance-aware segmentation**  
   Replace semantic segmentation + connected components with an architecture that explicitly models individual filaments.

2. **Boundary-aware learning**  
   Add boundary or distance-transform supervision to improve separation of touching structures.

3. **Probabilistic multi-annotator modelling**  
   Explicitly model uncertainty and disagreement between annotation records.

4. **Higher-resolution inference**  
   Use tiled 2048-resolution processing to better preserve small filaments.

5. **Cross-validation over date blocks**  
   Reduce dependence on a single validation split and obtain more robust estimates.

---

# 15. Acknowledgements

This project uses GONG H-alpha observations and the MAGFiLO dataset provided for the Solar Filament Segmentation Challenge 2026.

The challenge is supported by the U.S. National Science Foundation and the National Solar Observatory.

The solution was developed using PyTorch and Kaggle GPU infrastructure.

AI-assisted tools were used during parts of the implementation and documentation process. All major methodological decisions, experiments, evaluation procedures, and final results were reviewed by the author.

---

## Author

**Parsa Behjat Tabrizi**

Computer Science / Computer Engineering

Islamic Azad University, Pardis Branch

---

## References

- Ahmadzadeh et al., *A dataset of manually annotated filaments from H-alpha observations*, Scientific Data, 2024.
- Ahmadzadeh et al., *Solar Filament Segmentation Challenge 2026*, Kaggle.
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI, 2015.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR, 2016.
- Ioffe & Szegedy, *Batch Normalization*, ICML, 2015.
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR, 2015.
- Kirillov et al., *Panoptic Segmentation*, CVPR, 2019.
- Milletari et al., *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation*, 3DV, 2016.
