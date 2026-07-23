# Facial Expression Recognition with CNN

> **A convolutional neural network trained on the FER2013 dataset to classify facial expressions into 7 emotions from 48×48 grayscale face images.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/)
[![FER2013](https://img.shields.io/badge/Dataset-FER2013-8B5CF6)](https://www.kaggle.com/datasets/msambare/fer2013)
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

This notebook serves three purposes:

1. **A working FER2013 baseline** that can be extended with deeper architectures, transfer learning, or attention modules
2. **A teaching reference** for anyone learning CNNs, showing every design decision (architecture, augmentation, callbacks, metrics) with clear justifications
3. **A foundation** for downstream affective computing work such as real-time emotion recognition, mental health sentiment systems, and HCI research

---

## The Seven Emotions

The model predicts one of seven discrete emotion classes, the standard Ekman-inspired set used across most facial expression datasets:

| Label | Emotion | Typical Sample Size (FER2013) |
|-------|---------|-------------------------------|
| 0 | **Angry** | ~3,995 |
| 1 | **Disgust** | ~436 (minority) |
| 2 | **Fear** | ~4,097 |
| 3 | **Happy** | ~7,215 (majority) |
| 4 | **Neutral** | ~4,965 |
| 5 | **Sad** | ~4,830 |
| 6 | **Surprise** | ~3,171 |

Notice the class imbalance: Happy has roughly **17× more samples than Disgust**. This is a defining characteristic of FER2013 and directly shapes the training strategy, evaluation metrics, and the interpretation of confusion matrix patterns.

---

## Dataset: FER2013

**Name:** FER2013 (Facial Expression Recognition 2013)
**Origin:** Introduced for the ICML 2013 Challenges in Representation Learning
**Total images:** ~35,887
**Resolution:** 48×48 pixels, grayscale
**Kaggle path:** `/kaggle/input/fer2013/`

The dataset is pre-partitioned into two directories:

```
fer2013/
├── train/     # ~28,709 images — used for training and validation
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/      # ~7,178 images — held out for final evaluation
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

- **Train** (80% of `/train`) — used to fit the model
- **Validation** (20% of `/train`) — used for callbacks and early stopping
- **Test** (full `/test` directory) — held out for final evaluation

A sanity check at Step 5 verifies there is no file overlap between validation and test sets.

### Known Dataset Caveats

FER2013 is an imperfect but widely-used benchmark. Known issues:

- A non-trivial number of mislabelled images (~3–5% estimated)
- Low resolution (48×48) that limits achievable accuracy ceiling
- Class imbalance skewed toward Happy and away from Disgust
- Some non-face images that slipped through web scraping
- Human performance on FER2013 is approximately **65% accuracy**, which places a realistic upper bound on any model

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

Total parameters: approximately **4.5M trainable**, all fitting comfortably in a free Kaggle T4 GPU.

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Input shape** | 48 × 48 × 1 (grayscale) | FER2013 native resolution and colour depth |
| **Batch size** | 64 | Standard for mid-size CNNs on T4 GPU |
| **Max epochs** | 55 | Generous upper bound, early stopping usually halts sooner |
| **Optimiser** | Adam (default lr=0.001) | Reliable default for image classification from scratch |
| **Loss** | categorical_crossentropy | Standard for multi-class one-hot targets |
| **Validation split** | 0.2 (20%) | Stratified automatically via `ImageDataGenerator` |
| **EarlyStopping** | patience=8, restore_best_weights=True | Stops training if val_loss doesn't improve for 8 epochs |
| **ReduceLROnPlateau** | patience=4, factor=0.2 | Cuts learning rate by 5× if val_loss stalls for 4 epochs |

### Data Augmentation

Applied to the training set only (never to validation or test):

- `rotation_range=20` — ±20° random rotation
- `zoom_range=0.2` — random zoom up to 20%
- `horizontal_flip=True` — random horizontal flip (faces are approximately symmetric so this doubles effective data)
- `rescale=1./255` — normalise pixel values to [0, 1]

No vertical flip (upside-down faces aren't a realistic augmentation), no colour jitter (images are grayscale), no brightness adjustment (FER2013 already has wide brightness variation).

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

If the overlap count is not zero, the split is broken and metrics would be invalid. This check ensures the test set is genuinely held out.

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

**What it does:** Runs `model.fit()` for up to 55 epochs with both callbacks active. Training typically converges within 30–45 epochs on a T4 GPU.

A hyperparameter audit cell follows immediately after, printing every choice (batch size, image size, epochs, optimiser, augmentation settings, callback patience, architecture summary) so the notebook is self-documenting when shared or published.

---

### Step 9: Training Performance Analysis

**What it does:** Plots a **three-panel training performance figure**:

1. **Accuracy over epochs** — training vs validation accuracy curves
2. **Loss over epochs** — training vs validation loss curves
3. **Overfitting gap** — training accuracy minus validation accuracy, with a zero-reference line

The third panel is the tell-tale overfitting indicator. A healthy model keeps the gap small throughout training. A model that overfits shows the gap widening after the midpoint.

A training summary prints the best epoch (lowest val_loss), final train accuracy, final val accuracy, and final loss values.

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

---

### Step 12: Qualitative Inference on 200 Random Images

**What it does:** Samples 200 random images from the test set (without replacement, seeded at 42 for reproducibility) and produces a **10×20 grid** showing each image with:

- Its true class label
- The model's predicted label
- The prediction confidence percentage
- A green title for correct predictions, red for incorrect

At the end, a summary prints how many of the 200 were correctly predicted.

This qualitative check reveals failure patterns that aggregate metrics can hide: are the errors distributed across classes or concentrated on one pair? Are high-confidence predictions usually right? Do low-resolution images consistently fool the model?

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

---

## Results

> *Fill this section with your actual numbers after running the notebook. Placeholders below show the expected format.*

### Test Set Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | *(run notebook)* |
| **Macro F1** | *(run notebook)* |
| **Weighted F1** | *(run notebook)* |
| **Macro Precision** | *(run notebook)* |
| **Macro Recall** | *(run notebook)* |

### Per-Class F1

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|------|---------|
| Angry | | | | |
| Disgust | | | | |
| Fear | | | | |
| Happy | | | | |
| Neutral | | | | |
| Sad | | | | |
| Surprise | | | | |

### Context: FER2013 Benchmark Results

For reference, notable FER2013 results in the literature:

- **Human-level accuracy:** ~65%
- **Typical CNN from scratch:** 60–68%
- **Transfer learning (VGG, ResNet pretrained on ImageNet):** 68–72%
- **State-of-the-art (ensemble + advanced augmentation):** ~75%

Landing above 65% with a from-scratch CNN is a competitive baseline for this dataset.

### Common Error Patterns

On FER2013, virtually every model confuses certain emotion pairs at higher-than-random rates:

- **Fear ↔ Surprise** (visually similar wide-eyed expressions)
- **Sad ↔ Neutral** (subtle mouth shape differences at 48×48)
- **Angry ↔ Disgust** (similar brow furrow patterns)
- **Happy** is almost always the easiest class (distinct smile shape)
- **Disgust** is almost always the hardest class (minority + subtle features)

Your confusion matrix will almost certainly show these patterns.

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

### Option 2: Run Locally

```bash
# Clone the repo
git clone https://github.com/[your-username]/facial-expression-recognition-cnn.git
cd facial-expression-recognition-cnn

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
- Class imbalance (Happy has ~17× more samples than Disgust)
- Dataset contains approximately 3–5% mislabelled images
- Not demographically balanced across age, gender, or ethnicity
- Faces were web-scraped with limited consent metadata

**Ethical considerations for deployment:**

- **Emotion inference is not ground truth.** A model trained on posed or captured facial expressions predicts the *visual pattern* of an emotion, not the actual emotional state of the person. Context, culture, and individual variation make this a probabilistic inference at best
- **Cultural bias.** Facial expression datasets skew Western in both capture and labelling conventions. Deploying a FER2013-trained model on a culturally distinct population without validation is problematic
- **Consent and privacy.** Using facial expression inference in workplaces, schools, or public spaces raises significant privacy concerns and is regulated differently across jurisdictions (GDPR, CCPA, various AI Act provisions)
- **Fairness.** Before any real-world deployment, the model should be audited for performance disparities across demographic groups

This notebook is intended for **research, education, and academic benchmarking**. It is not production-ready for any consequential deployment without substantial additional work on bias auditing, demographic fairness, and user consent frameworks.

---

## Roadmap

Future improvements that may land in later versions:

- **Transfer learning baselines** — VGG16, ResNet50, EfficientNet-B0 pretrained on ImageNet
- **Attention modules** — CBAM or SE blocks to improve focus on emotion-relevant facial regions
- **Test-time augmentation (TTA)** — averaging predictions across augmented views for a 1–2 point accuracy boost
- **Model ensembling** — combining predictions from 3–5 independently trained models
- **Face detection preprocessing** — integrate MTCNN or MediaPipe to crop faces more tightly before inference
- **Real-time webcam demo** — OpenCV-based live inference script
- **Grad-CAM visualisations** — showing which face regions drive each prediction
- **FER+ dataset support** — FER2013's better-labelled successor
- **Multi-task learning** — joint emotion + valence/arousal prediction
- **Knowledge distillation** — distilling the CNN into a smaller mobile-deployable model

---

## Author

**Collins Lemeke**

This project was built as part of a wider research interest in efficient, accessible computer vision and affective computing. Facial expression recognition connects directly to my other work on lightweight NLP for mental health sentiment analysis and carbon-aware model design.

For questions, feedback, or feature requests, open a GitHub issue.

---

## License

MIT License. Free to use, modify, and distribute. See [LICENSE](LICENSE) for full terms.

The FER2013 dataset has its own licence and terms of use, separate from this code. Please refer to the [original Kaggle dataset page](https://www.kaggle.com/datasets/msambare/fer2013) for dataset licensing details.

---

> *Built with TensorFlow, Keras, and a lot of careful attention to the quirks of training from scratch on a small, imbalanced, low-resolution dataset.*
