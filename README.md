# Facial Expression Recognition with CNN

> **A convolutional neural network trained on the FER2013 dataset to classify facial expressions into 7 emotions from 48×48 grayscale face images.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/code/collinslemeke/facial-expression-recognition-with-cnn)
[![FER2013](https://img.shields.io/badge/Dataset-FER2013-8B5CF6)](https://www.kaggle.com/datasets/msambare/fer2013)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-66.15%25-success)](#results)
[![Macro F1](https://img.shields.io/badge/Macro%20F1-0.620-success)](#results)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [The Seven Emotions](#the-seven-emotions)
- [Dataset: FER2013](#dataset-fer2013)
- [CNN Architecture](#cnn-architecture)
- [Training Configuration](#training-configuration)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Step 1: Environment Setup and Data Quality Check](#step-1-environment-setup-and-data-quality-check)
  - [Step 2: Data Generators and Preprocessing](#step-2-data-generators-and-preprocessing)
  - [Step 3: Sample Visualisation](#step-3-sample-visualisation)
  - [Step 4: Data Augmentation](#step-4-data-augmentation)
  - [Step 5: Dataset Split Verification](#step-5-dataset-split-verification)
  - [Step 6: CNN Model Architecture](#step-6-cnn-model-architecture)
  - [Step 7: Compile and Callbacks](#step-7-compile-and-callbacks)
  - [Step 8: Model Training](#step-8-model-training)
  - [Step 9: Training Performance Analysis](#step-9-training-performance-analysis)
  - [Step 10: Confusion Matrix and Metrics](#step-10-confusion-matrix-and-metrics)
  - [Step 11: Test Set Evaluation](#step-11-test-set-evaluation)
  - [Step 12: Qualitative Inference on 200 Random Images](#step-12-qualitative-inference-on-200-random-images)
- [Key Design Decisions](#key-design-decisions)
- [Results](#results)
- [How to Reproduce](#how-to-reproduce)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Limitations and Ethical Considerations](#limitations-and-ethical-considerations)
- [Roadmap](#roadmap)
- [Author](#author)
- [License](#license)

---

## Overview

This project trains a **custom Convolutional Neural Network (CNN)** from scratch to classify facial expressions into **seven emotional categories** using the FER2013 dataset, a standard benchmark in affective computing and computer vision. The entire pipeline runs in a single Kaggle notebook, from raw dataset loading through to evaluation and qualitative inference on unseen test images.

The network is a classic three-block VGG-style architecture with batch normalisation, dropout regularisation, and a small dense head. It is trained end-to-end on 48×48 grayscale images with aggressive data augmentation to handle the class imbalance and limited resolution that make FER2013 a notoriously hard benchmark.

The headline result is **66.15% test accuracy**, which sits inside the 65 ± 5% band reported for human annotators on this benchmark. The more useful number is **macro F1 of 0.620**, and the gap between the two is the point of this repository: on a dataset this skewed, an aggregate score conceals where the model actually fails. Every evaluation choice here is designed to surface that.

This notebook serves three purposes:

1. **A working FER2013 baseline** that can be extended with deeper architectures, transfer learning, or attention modules
2. **A teaching reference** for anyone learning CNNs, showing every design decision (architecture, augmentation, callbacks, metrics) with clear justifications
3. **A foundation** for downstream affective computing work such as real-time emotion recognition, mental health sentiment systems, and HCI research

---

## The Seven Emotions

The model predicts one of seven discrete emotion classes, the standard Ekman-inspired set used across most facial expression datasets:

| Label | Emotion | Train Samples | Test Samples | Ratio to Smallest |
|-------|---------|---------------|--------------|-------------------|
| 0 | **Angry** | 3,995 | 958 | 9.16 |
| 1 | **Disgust** | 436 (minority) | 111 | 1.00 |
| 2 | **Fear** | 4,097 | 1,024 | 9.40 |
| 3 | **Happy** | 7,215 (majority) | 1,774 | 16.55 |
| 4 | **Neutral** | 4,965 | 1,233 | 11.39 |
| 5 | **Sad** | 4,830 | 1,247 | 11.08 |
| 6 | **Surprise** | 3,171 | 831 | 7.27 |
| | **Total** | **28,709** | **7,178** | |

Notice the class imbalance: Happy has roughly **16.55× more training samples than Disgust**, and the test partition is skewed almost identically at 15.98×. This is a defining characteristic of FER2013 and it directly shapes the training strategy, the evaluation metrics, and the interpretation of confusion matrix patterns.

**How much can that imbalance hide?** Because the test class proportions are public and fixed, the answer can be computed exactly rather than argued. Overall accuracy is the support-weighted mean of per-class recall, so the maximum accuracy attainable while a set of classes fails completely is `1 − (their combined share of the test set)`. Applying that to the three smallest classes — Angry, Disgust and Surprise — gives **73.53%**. A model that recognises no anger, no disgust and no surprise would still post 73.53%, which exceeds the 73.28% published for a tuned VGGNet on this benchmark. That is why this repository reports macro F1 and balanced accuracy alongside the headline figure.

---

## Dataset: FER2013

**Name:** FER2013 (Facial Expression Recognition 2013)
**Origin:** Introduced for the ICML 2013 Challenges in Representation Learning
**Total images:** 35,887
**Resolution:** 48×48 pixels, grayscale
**Kaggle path:** `/kaggle/input/fer2013/`

The dataset is pre-partitioned into two directories:

```
fer2013/
├── train/     # 28,709 images — used for training and validation
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/      # 7,178 images — held out for final evaluation
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

### Split Strategy

This notebook uses a **three-way split**:

- **Train** (80% of `/train`) — 22,968 images, used to fit the model
- **Validation** (20% of `/train`) — 5,741 images, used for callbacks and early stopping
- **Test** (full `/test` directory) — 7,178 images, held out for final evaluation

That is 64.0 / 16.0 / 20.0% of the full dataset. The split is applied **within each class directory**, so class proportions are preserved in both streams and validation is not distributionally different from training. Validation and test images originate from physically distinct source directories, which makes disjointness structural rather than probabilistic. A sanity check at Step 5 verifies programmatically that the file overlap is zero.

### Known Dataset Caveats

FER2013 is an imperfect but widely-used benchmark. Known issues:

- A non-trivial number of mislabelled images (~3–5% estimated)
- Low resolution (48×48) that limits achievable accuracy ceiling
- Class imbalance skewed toward Happy and away from Disgust
- Some non-face images that slipped through web scraping
- Human performance on FER2013 is approximately **65 ± 5% accuracy**, which places a realistic upper bound on any model
- **No demographic annotation whatsoever.** The dataset carries no age, gender or ethnicity labels, so the disparity metrics the fairness literature requires cannot be computed on it at all

These caveats are why this notebook reports **macro F1** alongside accuracy. Accuracy alone hides failure on minority classes.

---

## CNN Architecture

A three-block VGG-style CNN, trained from scratch on grayscale 48×48 inputs.

```
Input (48, 48, 1)
│
├── Block 1 ────────────────────────────────────
│   Conv2D(64, 3×3, padding=same, ReLU)
│   BatchNormalization
│   Conv2D(64, 3×3, padding=same, ReLU)
│   BatchNormalization
│   MaxPooling2D(2×2)      →  (24, 24, 64)
│   Dropout(0.25)
│
├── Block 2 ────────────────────────────────────
│   Conv2D(128, 3×3, padding=same, ReLU)
│   BatchNormalization
│   Conv2D(128, 3×3, padding=same, ReLU)
│   BatchNormalization
│   MaxPooling2D(2×2)      →  (12, 12, 128)
│   Dropout(0.25)
│
├── Block 3 ────────────────────────────────────
│   Conv2D(256, 3×3, padding=same, ReLU)
│   BatchNormalization
│   Conv2D(256, 3×3, padding=same, ReLU)
│   BatchNormalization
│   MaxPooling2D(2×2)      →  (6, 6, 256)
│   Dropout(0.25)
│
└── Classifier Head ────────────────────────────
    Flatten
    Dense(256, ReLU)
    BatchNormalization
    Dropout(0.5)
    Dense(7, Softmax)      →  Output (7 classes)
```

**Why this architecture?**

- **Progressive filter doubling (64 → 128 → 256):** Each block captures more abstract features as spatial dimensions shrink. Low-level edges in Block 1, local textures in Block 2, semantic emotion-relevant patterns in Block 3
- **Padding=same:** Keeps spatial dimensions stable within a block, so only MaxPool reduces them. This gives the network more layers at each spatial scale
- **BatchNorm after every Conv2D and Dense:** Accelerates training, stabilises gradients, and provides mild regularisation. Essential for training a CNN from scratch on a dataset this size
- **Dropout 0.25 after conv blocks, 0.5 before the final dense layer:** Heavier dropout is applied where overfitting risk is highest (the fully connected head). Lighter dropout in conv blocks because BatchNorm already regularises them
- **Dense(256) before the classifier:** Enough capacity to learn emotion-level abstractions without overwhelming the small 6×6×256 feature maps

**Parameter count:** 3,510,215 total, of which 3,507,911 are trainable and 2,304 are non-trainable batch normalisation statistics. All of it fits comfortably on a free Kaggle T4 GPU.

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Input shape** | 48 × 48 × 1 (grayscale) | FER2013 native resolution and colour depth |
| **Batch size** | 64 (359 steps per epoch) | Standard for mid-size CNNs on T4 GPU |
| **Max epochs** | 55 | Generous upper bound, early stopping usually halts sooner |
| **Optimiser** | Adam (initial lr = 1×10⁻³) | Reliable default for image classification from scratch |
| **Loss** | categorical_crossentropy | Standard for multi-class one-hot targets |
| **Validation split** | 0.2 (20%), applied per class directory | Preserves class proportions in both streams |
| **EarlyStopping** | patience=8 on val_loss, restore_best_weights=True | Stops training if val_loss doesn't improve for 8 epochs |
| **ReduceLROnPlateau** | patience=4, factor=0.2 | Cuts learning rate by 5× if val_loss stalls for 4 epochs |
| **Global seed** | 42, set before model construction | Fixes weight initialisation, shuffle order and augmentation draws |
| **Hardware** | Kaggle, NVIDIA Tesla T4 | Free tier |

### Data Augmentation

Applied to the training set only (never to validation or test):

- `rotation_range=20` — ±20° random rotation
- `zoom_range=0.2` — random zoom up to 20%
- `horizontal_flip=True` — random horizontal flip (faces are approximately symmetric so this doubles effective data)
- `rescale=1./255` — normalise pixel values to [0, 1]

No vertical flip (upside-down faces aren't a realistic deployment condition), no colour jitter (images are grayscale), no brightness adjustment (FER2013 already has wide brightness variation).

---

## Pipeline Walkthrough

The notebook executes 12 clearly numbered steps. Below is a detailed walkthrough of each one.

---

### Step 1: Environment Setup and Data Quality Check

**What it does:** Imports all required libraries (TensorFlow, NumPy, Matplotlib, Seaborn, PIL, scikit-learn) and runs a data quality inspection on both the train and test directories.

The `check_data_quality` function:
- Walks every class directory
- Opens each image via PIL and calls `img.verify()` to detect corruption
- Counts total images, corrupted images, and per-class sample counts
- Reports any corrupted file paths that need removal before training

This upfront check prevents silent errors mid-training when `ImageDataGenerator` hits a bad file.

---

### Step 2: Data Generators and Preprocessing

**What it does:** Configures three `ImageDataGenerator` instances for train, validation, and test.

The train generator reads from `/kaggle/input/fer2013/train` with `validation_split=0.1` initially (later re-configured to 0.2 in Step 4), grayscale colour mode, 48×48 target size, categorical class mode, and batch size 64. Images are rescaled from [0, 255] to [0, 1].

The test generator reads from `/kaggle/input/fer2013/test` with no augmentation and `shuffle=False` to preserve ordering for evaluation.

---

### Step 3: Sample Visualisation

**What it does:** Pulls a single batch from the train generator and displays 10 sample images in a 2×5 grid with their emotion labels.

This is a critical sanity check. You see the actual input the model will see (48×48 grayscale, possibly rotated or flipped by augmentation), and you can immediately spot:

- Any label-image mismatches
- The visual quality and resolution of the data
- How different emotions look at this low resolution (harder than you'd think — Fear and Surprise often look similar, Sad and Neutral often blur together)

---

### Step 4: Data Augmentation

**What it does:** Rebuilds the data generators with full augmentation enabled and `validation_split=0.2` for the final configuration used during training.

Augmentation is applied only to the training stream. Validation and test streams remain clean (just rescaling), so evaluation metrics reflect true generalisation, not augmented performance.

A split-visualisation cell then produces a **pie chart and stacked bar chart** showing the three-way split with per-set image counts and percentages, plus an annotation explaining that validation comes from `/train` (via split) while test comes from `/test` (native held-out).

---

### Step 5: Dataset Split Verification

**What it does:** Verifies split integrity programmatically:

- Prints validation set composition (source directory, total images, per-class distribution)
- Prints test set composition (same breakdown)
- Computes the **file overlap** between validation and test file paths. Must be zero

If the overlap count is not zero, the split is broken and metrics would be invalid. This check ensures the test set is genuinely held out. In this run the overlap is **0**, as expected given the two sets come from physically distinct directories.

---

### Step 6: CNN Model Architecture

**What it does:** Builds the Sequential Keras CNN described in the [Architecture section](#cnn-architecture) above and calls `model.summary()` to print the layer-by-layer parameter count.

---

### Step 7: Compile and Callbacks

**What it does:** Compiles the model with Adam optimiser, categorical cross-entropy loss, and accuracy metric. Then defines two training callbacks:

- **EarlyStopping** — monitors `val_loss`, stops training if no improvement over 8 epochs, restores the best weights from training
- **ReduceLROnPlateau** — monitors `val_loss`, reduces the learning rate by 5× (factor 0.2) if stalled for 4 epochs

This is the canonical callback combo for image classification. Early stopping prevents overfitting, plateau-based LR reduction helps the model escape local minima in the late stages.

---

### Step 8: Model Training

**What it does:** Runs `model.fit()` for up to 55 epochs with both callbacks active.

In this run, training halted at **epoch 34** when early stopping fired, with weights restored from **epoch 26** (the validation loss minimum, 0.9671). The plateau schedule reduced the learning rate three times: to 2×10⁻⁴ at epoch 14, 4×10⁻⁵ at epoch 25, and 8×10⁻⁶ at epoch 30. The first reduction produced the clearest single improvement in the run, with validation accuracy rising from 58.37% to 61.91% within one epoch, indicating the model had been oscillating rather than converging at the initial rate. Subsequent reductions yielded little further gain.

A hyperparameter audit cell follows immediately after, printing every choice (batch size, image size, epochs, optimiser, augmentation settings, callback patience, architecture summary) so the notebook is self-documenting when shared or published.

---

### Step 9: Training Performance Analysis

**What it does:** Plots a **three-panel training performance figure**:

1. **Accuracy over epochs** — training vs validation accuracy curves
2. **Loss over epochs** — training vs validation loss curves
3. **Overfitting gap** — training accuracy minus validation accuracy, with a zero-reference line

The third panel is the tell-tale overfitting indicator. A healthy model keeps the gap small throughout training. A model that overfits shows the gap widening after the midpoint.

At the selected epoch, training accuracy was **68.10%** against validation accuracy **64.17%**, a gap of **3.93 points**. The gap panel shows two early transients at epochs 2 and 7 before settling near four points, indicating the divergence was controlled rather than progressive — mild overfitting that early stopping caught before it widened.

---

### Step 10: Confusion Matrix and Metrics

**What it does:** Produces a comprehensive **four-panel evaluation figure** on the test set:

1. **Raw confusion matrix** (counts) — heatmap showing exact sample counts per true-predicted pair
2. **Normalised confusion matrix** (row percentages) — heatmap showing what percentage of each true class was predicted as each label
3. **Per-class F1 score bar chart** — with a horizontal line showing the macro-F1 average, so you can see which classes drag the mean down
4. **Overall metrics summary** — horizontal bar chart showing Accuracy, Macro F1, Weighted F1, Macro Precision, and Macro Recall

A full `classification_report` is printed below with per-class precision, recall, F1, and support.

**Why macro F1 matters:** Macro F1 averages F1 across all classes with equal weight, treating Disgust (minority) as equally important as Happy (majority). Accuracy and weighted F1 can both be gamed by ignoring minority classes. Macro F1 cannot.

---

### Step 11: Test Set Evaluation

**What it does:** Runs `model.evaluate(test_generator)` to produce the final headline test accuracy and loss numbers. Clean, simple, final.

Result: **66.15% accuracy, 0.9465 loss** over 7,178 held-out images.

---

### Step 12: Qualitative Inference on 200 Random Images

**What it does:** Samples 200 random images from the test set (without replacement, seeded at 42 for reproducibility) and produces a **10×20 grid** showing each image with:

- Its true class label
- The model's predicted label
- The prediction confidence percentage
- A green title for correct predictions, red for incorrect

At the end, a summary prints how many of the 200 were correctly predicted.

**In this run, 118 of 200 were correct — an agreement rate of 59.0% (95% CI 52.3–65.7%).** That is lower than the 66.15% aggregate, and the discrepancy is worth stating rather than glossing over. Under hypergeometric sampling from a population of 7,178 images containing 4,748 correct predictions, the expected count is 132.3 with a standard deviation of 6.60, placing the observed value **2.17 standard errors below the aggregate** (p ≈ 0.03).

In plain terms: the audit sample ran modestly unlucky. The consequence is specific — the grid's apparent error rate should not be read as the model's error rate, for which the 7,178-image aggregate is the authoritative estimate. Had the grid been displayed without this check, a reader forming an impression of roughly 59% performance would have formed a false one.

This is also an argument against the small qualitative panels common in the applied FER literature. At n=200 the confidence interval already spans 13.4 points; at the panel sizes typically published it is wider still, and without a representativeness test the reader has no way to know.

Two patterns are visible in the grid itself. High-confidence errors cluster on the class pairs the confusion matrix identifies rather than distributing uniformly, indicating systematic rather than random failure. And a subset of misclassified images appear arguably mislabelled in the source data, consistent with FER2013's known label noise.

---

## Key Design Decisions

A handful of deliberate choices make this baseline strong despite FER2013's difficulty.

| Decision | Choice | Why |
|----------|--------|-----|
| **Architecture** | Three-block VGG-style CNN, trained from scratch | Simpler than ResNet or EfficientNet, easier to explain in a teaching context, and still reaches competitive FER2013 numbers |
| **Colour mode** | Grayscale only (1 channel) | FER2013 is already grayscale. Using 3-channel input would waste parameters on a synthetic colour expansion |
| **Input size** | 48 × 48 native | Upscaling to 224×224 would not add signal, it would just add compute. The information ceiling is set by the original capture resolution |
| **BatchNorm everywhere** | After every Conv2D and Dense (except output) | Critical for training-from-scratch stability on a modestly-sized dataset |
| **Dropout schedule** | 0.25 after conv blocks, 0.5 before the classifier | Heavier regularisation at the FC head where parameters and overfitting risk concentrate |
| **No class weights** | Not used, despite strong imbalance | Augmentation + BatchNorm + dropout provide sufficient regularisation. Class weights were tested separately and introduced instability without meaningful F1 improvement |
| **Best metric** | Macro F1 (primary), accuracy (secondary) | Macro F1 honestly reflects minority class (Disgust, Fear) performance |
| **Early stopping metric** | val_loss, not val_accuracy | Loss is smoother and less noisy than accuracy on an imbalanced dataset |
| **LR reduction** | Factor 0.2, patience 4 | Aggressive 5× cut triggered before early stopping kicks in. Gives the model a chance to recover before giving up |
| **Seed** | 42 for train/val sampling and the 200-image inference grid | Reproducibility for paper submission and dissertation appendices |
| **Audit representativeness test** | Computed before interpreting the 200-image grid | An audit reported without a representativeness test is a rhetorical device, not evidence |

---

## Results

All figures below were read from the stored outputs of the published notebook, or derived arithmetically from the classification report and the class counts above.

### Test Set Performance

Evaluated on the full held-out test directory, 7,178 images never touched during training, hyperparameter selection or early stopping.

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **66.15%** |
| 95% Confidence Interval | [65.06%, 67.24%] |
| Test Loss | 0.9465 |
| **Macro F1** | **0.6202** |
| Weighted F1 | 0.6572 |
| Macro Precision | 0.6696 |
| Macro Recall (balanced accuracy) | 0.6049 |

### The Cost of Aggregation

| Quantity | Value |
|----------|-------|
| Accuracy − balanced accuracy | **5.66 points** |
| Accuracy − macro F1 | 4.13 points |
| Weighted F1 − macro F1 | 0.037 |

Balanced accuracy weights every class equally while accuracy weights by support, so the 5.66-point difference is precisely the portion of the headline figure attributable to the test distribution rather than to competence spread evenly across the task. That gap is **wider than the 95% confidence interval on the accuracy estimate itself**, which is the clearest possible argument for reporting both.

### Generalisation Check

| Quantity | Value |
|----------|-------|
| Validation accuracy (epoch 26) | 64.17% |
| Test accuracy | 66.15% |
| **Test − validation** | **+1.98 points** |
| Training accuracy (epoch 26) | 68.10% |
| Training − validation | 3.93 points |

The **sign** of the test-minus-validation difference is the thing to look at. Contamination of a test estimate through model selection manifests as validation *optimism*: a selected checkpoint looks better on the data used to select it than on data it has never influenced. Here the opposite is observed. The validation estimate was conservative rather than inflated, which is the signature of a protocol in which model selection could not reach the test partition.

### Per-Class Performance

| Emotion | Precision | Recall | F1 | Test Support | Train Support |
|---------|-----------|--------|------|--------------|---------------|
| Angry | 0.539 | 0.635 | 0.583 | 958 | 3,995 |
| Disgust | 0.796 | 0.351 | 0.488 | 111 | 436 |
| Fear | 0.579 | 0.371 | **0.452** | 1,024 | 4,097 |
| Happy | 0.866 | 0.870 | **0.868** | 1,774 | 7,215 |
| Neutral | 0.575 | 0.696 | 0.630 | 1,233 | 4,965 |
| Sad | 0.541 | 0.553 | 0.547 | 1,247 | 4,830 |
| Surprise | 0.790 | 0.758 | 0.774 | 831 | 3,171 |
| **Macro avg** | **0.670** | **0.605** | **0.620** | **7,178** | **28,709** |
| **Weighted avg** | 0.665 | 0.662 | 0.657 | 7,178 | 28,709 |

The spread is the substantive finding. Happy reaches F1 0.868 and Surprise 0.774, both well above the aggregate. At the other end, Fear attains 0.452 and Disgust 0.488, with Disgust recall at 0.351 — meaning roughly **two-thirds of disgust images are missed**. Reporting accuracy alone would overstate this model's class-balanced capability by more than five points.

Note also that the entire difference between this model's Disgust performance and *complete failure on the class* is 0.55 accuracy points. Disgust is simultaneously the model's clearest fairness deficit and the one least visible in its headline number.

### Two Failure Mechanisms, Not One

It is tempting to attribute all minority failure to scarcity of training data. The measurements do not support so simple a story. Rank correlation between training support and per-class recall is ρ = 0.536 (p = 0.215, n = 7) — a moderate association that does not reach significance, and which two classes contradict outright. Surprise is the second smallest class yet attains the second highest recall. Fear is the third largest yet attains the second lowest.

The **precision–recall asymmetry** separates the two mechanisms:

| Class | Times Predicted | True Count | Pred / True | Mechanism |
|-------|-----------------|------------|-------------|-----------|
| Disgust | 49 | 111 | 0.44 | **Under-prediction (scarcity)** |
| Fear | 656 | 1,024 | 0.64 | **Confusability** |
| Surprise | 797 | 831 | 0.96 | Balanced |
| Happy | 1,782 | 1,774 | 1.00 | Balanced |
| Sad | 1,273 | 1,247 | 1.02 | Balanced |
| Angry | 1,129 | 958 | 1.18 | Over-prediction |
| Neutral | 1,492 | 1,233 | 1.21 | Over-prediction |
| **Total** | **7,178** | **7,178** | **1.00** | |

**Disgust — scarcity.** The model emits the label only 49 times where 111 images carry it, under-predicting by more than half, but is right on four-fifths of those emissions. Low recall with high precision is the classical signature of scarcity: the decision boundary has been drawn conservatively because the class contributed little to the loss, so the model commits to it only when the evidence is strong. This is the failure mode class weighting and focal loss are designed to correct.

The confusion matrix sharpens the account. Of the 72 misclassified Disgust images, **51 (70.8%) are assigned to Angry alone**. As a proportion of the class, Disgust is labelled Angry **45.95%** of the time against a correct-label rate of 35.14% — so the model routes a disgust expression to Angry more often than it recognises it. The two categories are adjacent in both valence and facial action, and Angry carries nine times the training support, so the boundary between them sits well inside the region Disgust occupies. The displaced mass is visible on the other side of the ledger: Angry accumulates 521 false positives and is over-predicted at 1.18. Scarcity does not merely suppress a class; it determines which neighbour absorbs it.

**Fear — confusability.** Fear behaves differently. It is under-predicted (0.64) but its precision is also low (0.579), so the errors are bidirectional: the model both misses Fear images and misapplies the label. With 4,097 training images available, scarcity cannot be the explanation. The confusion structure shows the mass displaced toward Sad (22.66%), Angry (15.53%) and Neutral (12.99%), spread across three destinations rather than concentrated in one, which is consistent with genuine visual ambiguity at 48×48 resolution rather than with insufficient data.

**The practical consequence:** a single mitigation would not address both. Rebalancing would likely improve Disgust while leaving Fear largely untouched, since the latter requires either higher input resolution or representations that better separate confusable expressions. Measurement of this kind is what makes the distinction visible; an aggregate accuracy figure would not have separated them.

### Where the Errors Actually Went

Row-normalised confusion matrix, correct-class recall on the diagonal:

| True ↓ / Predicted → | Angry | Disgust | Fear | Happy | Neutral | Sad | Surprise |
|----------------------|-------|---------|------|-------|---------|-----|----------|
| **Angry** | **63.47%** | 0.63% | 6.58% | 2.92% | 12.00% | 12.42% | 1.98% |
| **Disgust** | 45.95% | **35.14%** | 5.41% | 1.80% | 3.60% | 7.21% | 0.90% |
| **Fear** | 15.53% | 0.00% | **37.11%** | 3.03% | 12.99% | 22.66% | 8.69% |
| **Happy** | 2.42% | 0.11% | 1.18% | **87.03%** | 5.75% | 2.09% | 1.41% |
| **Neutral** | 5.76% | 0.00% | 3.65% | 6.16% | **69.59%** | 13.46% | 1.38% |
| **Sad** | 12.75% | 0.16% | 6.34% | 3.93% | 20.29% | **55.25%** | 1.28% |
| **Surprise** | 4.57% | 0.00% | 7.46% | 6.26% | 3.25% | 2.65% | **75.81%** |

The largest single off-diagonal proportion is **Disgust → Angry at 45.95%**, which exceeds Disgust's own recall of 35.14%. Remaining concentrations are Fear → Sad (22.66%), Sad → Neutral (20.29%), Fear → Angry (15.53%) and Neutral → Sad (13.46%) — pairs whose distinguishing cues are least resolvable at 48×48.

### Context: FER2013 Benchmark Results

| System | Approach | Accuracy |
|--------|----------|----------|
| Uniform random baseline | Distribution property | 14.29% |
| Majority-class baseline | Distribution property | 24.71% |
| **This work — balanced accuracy** | *class-balanced view* | **60.49%** |
| Human annotators (Goodfellow et al.) | — | 65 ± 5% |
| **This work — accuracy** | 3-block CNN, from scratch | **66.15%** |
| Tang (2013) | CNN with L2-SVM objective | 71.16% |
| Khaireddin & Chen (2021) | Tuned VGGNet | 73.28% |
| *Blindness ceiling* | *3 of 7 classes at zero* | *73.53%* |
| Pramerdorfer & Kampel (2016) | CNN ensemble | 75.2% |

This model sits below the ensembled and heavily tuned systems, which is the honest position for a from-scratch model of this size without pretraining. The 66.15% figure lies wholly within the 65 ± 5% band reported for human annotators on this benchmark, which is the appropriate comparison rather than claiming parity from a point estimate.

The table also makes the blindness ceiling concrete: it falls **between two published results**, so a model recognising none of three emotions would outrank a genuine state-of-the-art system in any league table ordered by accuracy.

### What the Evidence Does Not Support

Several limits deserve statement rather than burial.

- **Support does not fully explain minority failure.** The rank correlation between training support and recall is ρ = 0.536 with p = 0.215, which with seven classes is a weak test. The two-mechanism account is presented as an interpretation supported by the precision–recall pattern and the confusion structure, not as a statistically established causal claim
- **Fairness here is class-level, not demographic.** FER2013 carries no age, gender or ethnicity annotation, so the disparity metrics the fairness literature requires cannot be computed at all. A model distributing error evenly across seven emotion classes could still distribute it very unevenly across demographic groups, and nothing here would detect that
- **This is a single run.** One seeded training run, therefore no variance estimate. Seeding makes the run reproducible; it does not make it representative of the seed distribution. The reported interval reflects test-set sampling error alone. A mean and standard deviation over several seeds would be the stronger claim
- **The 200-image audit diverged from the aggregate.** At 2.17 standard errors the sample is marginally unrepresentative, and its use has been restricted accordingly
- **The label ceiling is external to the model.** FER2013's known label noise means part of the residual error is irreducible, limiting what any architecture can demonstrate here and arguing for validation on relabelled or in-the-wild successors such as FER+ or AffectNet

---

## How to Reproduce

### Option 1: Run on Kaggle (Recommended)

1. Open [Kaggle](https://www.kaggle.com/) and sign in
2. Create a new notebook
3. Attach the FER2013 dataset from the Kaggle data tab: search for `msambare/fer2013` and click **Add**
4. Upload `facial-expression-recognition-with-cnn.ipynb` or copy the code cells
5. Enable **GPU T4 x1** in notebook settings (free tier)
6. Run all cells top to bottom

Total runtime: approximately 30–50 minutes on a T4 GPU.

The global seed is fixed at 42 before model construction, and the train/validation split enumerates each class directory as a lexicographically sorted file list and slices by index. An independent party running the same code on the same data therefore obtains the same partition.

### Option 2: Run Locally

```bash
# Clone the repo
git clone https://github.com/CollinsLemeke/Facial-Expression-Recognition-Model.git
cd Facial-Expression-Recognition-Model

# Install dependencies
pip install -r requirements.txt

# Download FER2013 dataset from Kaggle
# (You'll need a Kaggle API token — see https://www.kaggle.com/docs/api)
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/

# Update the paths in the notebook from /kaggle/input/fer2013 to data/fer2013
# Then run the notebook
jupyter notebook facial-expression-recognition-with-cnn.ipynb
```

### Hardware Recommendations

- **Minimum:** CPU-only. Training will take 6–10 hours. Not recommended
- **Recommended:** Any single modern GPU (T4, RTX 3060+, A10G). 30–50 minute training
- **Best:** A100 or L4. 15–25 minute training

---

## Repository Structure

```
.
├── README.md                                            # This file
├── facial-expression-recognition-with-cnn.ipynb        # Complete training notebook
├── requirements.txt                                     # Python dependencies
├── outputs/                                             # (generated)
│   ├── data_split_visualisation.png
│   ├── training_performance.png
│   ├── comprehensive_evaluation.png
│   └── 200_test_predictions.png
└── LICENSE
```

The notebook is self-contained. Running it end-to-end produces all four output figures listed in `outputs/`, which are the plots you'd include in a paper or dissertation.

---

## Dependencies

```
tensorflow>=2.15.0
numpy>=1.26.0
pandas>=2.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0
Pillow>=10.0.0
```

Install with:

```bash
pip install -r requirements.txt
```

On Kaggle, all of these are pre-installed. No setup needed.

---

## Limitations and Ethical Considerations

Facial expression recognition is a sensitive domain and FER2013 comes with real limitations that should be acknowledged before any downstream use.

**Dataset limitations:**

- Low 48×48 resolution limits the maximum achievable accuracy
- Class imbalance (Happy has 16.55× more training samples than Disgust)
- Dataset contains approximately 3–5% mislabelled images
- Not demographically balanced across age, gender, or ethnicity, and carries no demographic annotation with which to check
- Faces were web-scraped with limited consent metadata

**Ethical considerations for deployment:**

- **Emotion inference is not ground truth.** A model trained on captured facial expressions predicts the *visual pattern* an annotator assigned to a facial configuration, not the actual emotional state of the person. Context, culture, and individual variation make this a probabilistic inference at best. A measured balanced accuracy of 60.49%, with fewer than two in five disgust expressions recognised, is not a reliable instrument for consequential decisions
- **Cultural bias.** Facial expression datasets skew Western in both capture and labelling conventions. Deploying a FER2013-trained model on a culturally distinct population without validation is problematic
- **Regulatory position.** Article 5(1)(f) of Regulation (EU) 2024/1689 prohibits placing on the market or using AI systems to infer emotions of a natural person in workplace and education settings, except for medical or safety reasons; the prohibition has applied since 2 February 2025. Two applications commonly cited as motivation for FER research — classroom engagement monitoring and workplace affect analytics — therefore fall within a prohibited category in the EU, and this model must not be deployed for them
- **Fairness.** Before any real-world deployment, the model should be audited for performance disparities across demographic groups. FER2013 forecloses that check entirely

This notebook is intended for **research, education, and academic benchmarking**. It is not production-ready for any consequential deployment without substantial additional work on bias auditing, demographic fairness, and user consent frameworks.

---

## Roadmap

Future improvements that may land in later versions, ordered by what the results above actually justify:

- **Multi-seed evaluation** — replace single-run point estimates with distributions. This is the single most valuable next step
- **Class weighting or focal loss** — evaluated against the per-class baseline established here, with Fear serving as a control that should *not* respond if the two-mechanism account is correct
- **Higher-resolution inputs or attention modules** — CBAM or SE blocks, targeting the confusability mechanism rather than the scarcity one
- **Cross-dataset validation on demographically annotated corpora** — AffectNet or RAF-DB, to open the demographic fairness question FER2013 forecloses
- **Transfer learning baselines** — VGG16, ResNet50, EfficientNet-B0 pretrained on ImageNet
- **Test-time augmentation (TTA)** — averaging predictions across augmented views
- **Model ensembling** — combining predictions from 3–5 independently trained models
- **Face detection preprocessing** — integrate MTCNN or MediaPipe to crop faces more tightly before inference
- **Real-time webcam demo** — OpenCV-based live inference script
- **Grad-CAM visualisations** — showing which face regions drive each prediction, with the caveat that saliency localises evidence without establishing that the localised region is the model's operative reason
- **FER+ dataset support** — FER2013's better-labelled successor
- **Knowledge distillation** — distilling the CNN into a smaller mobile-deployable model

---

## Author

**Collins Lemeke**

AI Research Engineer, Centre of Intelligence of Things, University of Greater Manchester.

This project was built as part of a wider research interest in efficient, accessible computer vision and affective computing. Facial expression recognition connects directly to my other work on lightweight NLP for mental health sentiment analysis and carbon-aware model design.

- [Kaggle notebook](https://www.kaggle.com/code/collinslemeke/facial-expression-recognition-with-cnn)
- [GitHub](https://github.com/CollinsLemeke)

For questions, feedback, or feature requests, open a GitHub issue.

---

## License

MIT License. Free to use, modify, and distribute. See [LICENSE](LICENSE) for full terms.

The FER2013 dataset has its own licence and terms of use, separate from this code. Please refer to the [original Kaggle dataset page](https://www.kaggle.com/datasets/msambare/fer2013) for dataset licensing details.

---

> *Built with TensorFlow, Keras, and a lot of careful attention to the quirks of training from scratch on a small, imbalanced, low-resolution dataset.*
