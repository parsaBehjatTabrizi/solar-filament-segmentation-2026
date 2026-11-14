# Solar Filament Segmentation Challenge 2026 — U-Net with a pretrained ResNet-50 encoder

Solution to the **Solar Filament Segmentation Challenge 2026** (Kaggle: `filament-segmentation-2026`, organized by the Earth-Space AI Research Lab,
sponsored by the U.S. National Solar Observatory). The task is instance segmentation of solar filaments in GONG H-alpha images (2048 x 2048),
scored with a pooled panoptic quality (PQ) that uses many-to-many matching and compares the predictions of an image with **every annotator's record**.

**Final result:** public leaderboard **PQ 0.34**. The same configuration trained *without* the validation images scores **0.4435** on our date-grouped validation split
(official protocol). The leaderboard has been consistently about 0.09–0.10 below validation; every change we made moved both almost equally.

The 4-page technical report is in [`report/technical_report.pdf`](report/technical_report.pdf).

---

## Method in one paragraph

Semantic segmentation of the union of annotated filaments followed by a simple instance extraction.
A **U-Net with an ImageNet-pretrained ResNet-50 encoder** (32.6 M parameters) is trained at 1024 x 1024 resolution with BCE + soft Dice loss.
At every epoch each training image uses a **randomly chosen annotator record** as target (the metric scores every record separately).
BatchNorm running statistics are re-estimated after training (our first models showed unstable validation behaviour caused by stale statistics).
At inference the model is applied to the image and its horizontal / vertical / double flips (probabilities averaged), the map is up-sampled to 2048 x 2048,
thresholded at 0.5, a morphological **closing** (radius 10 px) re-joins fragments, **8-connected components** become the instances, and components of **at most 400 px** are dropped.
The final model is trained on **all 707 labelled images** (last epoch).

```
2048² image ─► resize 1024², 1–99 % normalisation ─► U-Net (ResNet-50), 4 flip views ─► mean probability
            ─► up-sample 2048² ─► threshold 0.5 ─► closing r=10 ─► connected components ─► drop ≤ 400 px ─► RLE ─► submission.csv
```

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── configs/final_config.yaml            hyper-parameters of the final model and of the inference
├── src/                                 reusable code (the notebooks contain the same logic)
│   ├── dataset.py                       image loading, annotation parsing, date-grouped split, training dataset
│   ├── model.py                         U-Net with ResNet encoder, weight loading
│   ├── losses.py                        BCE + soft Dice
│   ├── metrics.py                       competition metric (pooled PQ, many-to-many matching)
│   ├── postprocessing.py                threshold, closing, connected components, size filter
│   ├── rle.py                           run-length encoding / decoding
│   └── inference.py                     image -> probability map -> instances -> submission.csv (CLI)
├── tests/test_core.py                   unit tests (metric rules, post-processing, RLE)
├── notebooks/
│   ├── reproduce_submission.ipynb       ★ reproduces the submitted predictions from the stored weights (no training)
│   ├── 01_dataset_analysis_part1.ipynb  dataset audit / EDA
│   ├── 01_dataset_analysis_part2.ipynb
│   ├── 02_metric_and_evaluation.ipynb   official-protocol evaluation, cross-check with the organizers' scoring code
│   ├── 03_baseline_training.ipynb       Baseline-01: plain U-Net from scratch
│   ├── 04_experiments/                  all experiments (multi-annotator targets, pretrained encoders, ensembles, diagnostics ...)
│   └── 05_train_final_model.ipynb       training of the FINAL model on all 707 images
├── weights/final_model.pth              final weights (float16 state dict, ~65 MB)
├── results/
│   ├── experiment_results.csv           every experiment: validation PQ and public leaderboard PQ
│   ├── validation_metrics.csv           detailed validation metrics and error analysis of the train-only model
│   ├── training_curve.png               training / validation curves (ResNet-50 run)
│   ├── reproduction_run_gpu.ipynb       executed reproduction run (agreement with the submitted file: 0.9994)
│   └── qualitative_results/             example predictions
├── submission/submission.csv            final submitted predictions
└── report/technical_report.pdf          4-page technical report
```

The notebooks were run on **Kaggle** (one T4 GPU) and use Kaggle paths (`/kaggle/input/...`); the `src/` package works anywhere.

---

## Reproduce the submitted predictions (no training)

**Requirements:** Python 3.13 (tested on Kaggle), `pip install -r requirements.txt`. A GPU makes inference take minutes; on CPU it is much slower.
The competition data is **not** included (see the competition rules): download the test images from the Kaggle competition page.

**Option A — notebook (recommended, this is the pipeline notebook).** On Kaggle (GPU on, Internet on) add the competition data as input, run
`!git clone https://github.com/parsaBehjatTabrizi/solar-filament-segmentation-2026` in a cell (the notebook then finds `weights/final_model.pth` by itself) and *Run all* of
`notebooks/reproduce_submission.ipynb`. It writes `submission.csv` and, if the submitted file is available, prints the agreement with it.

**Option B — command line:**

```bash
pip install -r requirements.txt
python -m src.inference --weights weights/final_model.pth --test-dir /path/to/test_images --out submission.csv
pytest -q            # optional: unit tests of the metric, post-processing and RLE
```

**Reproduction check.** The stored weights are in half precision, so a few boundary pixels can change. In our run the reproduced file has 1,260 filaments over 177 test images
against 1,241 over 178 in the submitted file, and the pooled PQ between the two files is **0.9994** (every reproduced filament matches a submitted one and vice versa;
see `results/reproduction_run_gpu.ipynb`).

## Train from scratch

`notebooks/05_train_final_model.ipynb` (Kaggle, GPU on, **Internet on** to download the public torchvision ImageNet weights, about 2–3 hours on a T4) trains the final model on all 707 images
with the configuration in `configs/final_config.yaml`. `03_baseline_training.ipynb` and `notebooks/04_experiments/` contain the development runs on the 568 / 139 date-grouped split.
Re-training the same configuration changes the validation PQ by up to about ±0.01 (GPU non-determinism), so the weights are provided.

---

## Evaluation protocol and results

* **Metric:** pooled PQ with many-to-many matching (IoU strictly greater than 0.5 is a true positive, its IoU is summed in the numerator;
  unmatched predictions are false positives, unmatched ground-truth instances false negatives). The predictions of an image are scored separately against every
  annotator record and the counts are pooled. `src/metrics.py` re-implements this and agrees with the organizers' scoring code to six decimals on validation predictions.
* **Validation:** date-grouped 80 / 20 split (seed 42): 568 training / 139 validation images (225 annotator records); the 180 test images were never used for decisions.

| step | validation PQ | public LB |
|---|---:|---:|
| U-Net from scratch, threshold only | 0.290 | 0.21 |
| + epoch-50 weights, BN re-estimation, flip TTA | 0.304* | 0.22 |
| + remove objects ≤ 200 px | 0.351* | 0.26 |
| + remove ≤ 400 px, closing r = 10 ("recipe") | 0.372 | 0.28 |
| + random annotator record per epoch (B02a) | 0.384 | 0.29 |
| ResNet-34 pretrained encoder (B03) | 0.433 | 0.33 |
| ResNet-50 pretrained encoder (B04) | 0.433 (0.444 in a re-run) | 0.33 |
| **ResNet-50 trained on all 707 images (final)** | — (no held-out set) | **0.34** |

\* primary-record protocol (differs from the official protocol by at most 0.002 where both were computed). Full table: `results/experiment_results.csv`.

Things that did **not** help: averaging models (0.439 vs 0.4435 for the best single model), a deeper encoder, and instance-level confidence filtering (+0.005 on one half of the dates only).

**Error analysis** (train-only ResNet-50, validation; `results/validation_metrics.csv`): TP 1,063 / FP 577 / FN 544, mean IoU of matched pairs 0.677.
Most errors are borderline: 365 of 577 false positives overlap an annotated filament with IoU ≤ 0.5, and 220 of 544 misses overlap exactly one prediction with IoU ≤ 0.5.
Passing the exact ground-truth union through our pipeline gives PQ 0.83, so instance extraction is not the main limit. Small filaments (200–999 px) are hardest.
Annotators disagree as well (another annotator's record scores 0.38 PQ against the primary record on the 59 validation images with several records); we treat this as a reference level, not as a ceiling.

## Limitations

* The leaderboard score (0.34) is below the validation estimate; we could **not** attribute the gap to temporal distance (test images are on average slightly *closer* in time to training images than validation images).
  Label differences in the hidden test annotations and the small size of the public subset are candidate explanations that we could not test.
* The final model has no held-out validation score (all images were used for training); its evidence is the identical configuration that reached 0.4435 on validation and 0.33 on the leaderboard, plus the leaderboard score of 0.34.
* The semantic mask cannot separate touching filaments and there is no explicit boundary model; hyper-parameters were tuned on a single validation split.

## AI assistance

Parts of the code and of the technical report were written with the assistance of a large language model (Anthropic's Claude). All methodological choices and results were reviewed by the author.

## Data, license and acknowledgments

The competition data (MAGFiLO dataset) is not redistributed here. The code in this repository is released under the MIT license (see `LICENSE`); the encoder is initialised from the public torchvision ImageNet weights.
The organization of the challenge is supported by the U.S. National Science Foundation (Grants 2209912, 2433781, 2511630); the challenge is sponsored by the NSF National Solar Observatory.
This work uses GONG data obtained by the NSO Integrated Synoptic Program, managed by the National Solar Observatory, which is operated by AURA, Inc. under a cooperative agreement with the NSF.

## Contact

Parsa Behjat Tabrizi — behjattabrizi.sp@gmail.com — GitHub: [parsaBehjatTabrizi](https://github.com/parsaBehjatTabrizi)
