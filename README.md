# Facial Expression Recognition with CNN

> A convolutional neural network trained on the FER2013 dataset to sort 48x48 grayscale face images into 7 emotions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/code/collinslemeke/facial-expression-recognition-with-cnn)
[![FER2013](https://img.shields.io/badge/Dataset-FER2013-8B5CF6)](https://www.kaggle.com/datasets/msambare/fer2013)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-66.45%25-success)](#results-in-detail)
[![Macro F1](https://img.shields.io/badge/Macro%20F1-0.630-success)](#results-in-detail)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Table of Contents

- [Start here: what this project actually is](#start-here-what-this-project-actually-is)
- [Plain English glossary](#plain-english-glossary)
- [Results at a glance](#results-at-a-glance)
- [The seven emotions and the imbalance problem](#the-seven-emotions-and-the-imbalance-problem)
- [The dataset: FER2013](#the-dataset-fer2013)
- [The model](#the-model)
- [Training setup](#training-setup)
- [What the notebook does, cell by cell](#what-the-notebook-does-cell-by-cell)
- [What actually happened during training](#what-actually-happened-during-training)
- [Results in detail](#results-in-detail)
- [Why accuracy on its own would mislead you](#why-accuracy-on-its-own-would-mislead-you)
- [How this compares to published work](#how-this-compares-to-published-work)
- [What the evidence does not support](#what-the-evidence-does-not-support)
- [How to reproduce](#how-to-reproduce)
- [Repository structure](#repository-structure)
- [Dependencies](#dependencies)
- [Limitations and ethics](#limitations-and-ethics)
- [Roadmap](#roadmap)
- [Which run these numbers come from](#which-run-these-numbers-come-from)
- [Author](#author)
- [License](#license)

---

## Start here: what this project actually is

Imagine you have 35,887 small black and white photographs of faces. Each one has been given a label by a human: angry, disgusted, afraid, happy, neutral, sad, or surprised. The goal is to build a computer program that looks at a face it has never seen before and guesses which of those seven labels a human would have given it.

That is the whole project. This repository contains one Kaggle notebook that does it end to end:

1. Loads the images and checks none of them are broken
2. Splits them into three piles: one to learn from, one to check progress on, one kept locked away for the final exam
3. Builds a neural network from scratch
4. Trains it for up to 55 rounds
5. Measures how well it did, in several different ways, because one number is not enough
6. Shows 200 random test images with the model's guesses so you can see the behaviour with your own eyes

**The headline result is 66.45% correct on the 7,178 unseen test images.** That sounds mediocre until you learn that humans score roughly 65% plus or minus 5% on this same dataset. The images are only 48 by 48 pixels, some of the labels are wrong, and telling fear apart from sadness at that resolution is genuinely hard for anyone.

The more interesting number is **macro F1 of 0.630**, and the gap between those two figures is the real point of this repository. On a dataset where one emotion has 16 times more examples than another, a single overall score hides where the model is failing. Everything in the evaluation section is designed to expose that.

**Who this is for:**

- Anyone learning how a CNN is built and trained, step by step, with reasons given for every choice
- Anyone who wants a working FER2013 baseline to improve on
- Anyone who needs a worked example of honest model evaluation on imbalanced data

---

## Plain English glossary

If you are new to machine learning, these are the only terms you need to follow the rest of this README.

| Term | What it means in plain English |
|------|-------------------------------|
| **CNN (convolutional neural network)** | A type of neural network built for images. It scans small patches of the picture looking for patterns, starting with simple edges and building up to whole-face features. |
| **Training set** | The photos the model learns from. |
| **Validation set** | Photos held aside to check progress during training. The model never learns from these, but they are used to decide when to stop. |
| **Test set** | Photos locked away and used once, at the very end. This is the honest score. |
| **Epoch** | One complete pass through all the training photos. This model ran 55 of them. |
| **Accuracy** | Out of all test photos, what share did the model get right. Simple, but easily flattered by imbalanced data. |
| **Recall** | Out of all the truly angry faces, what share did the model catch. Answers "what is it missing?" |
| **Precision** | Out of all the faces the model called angry, what share really were angry. Answers "when it says angry, can I trust it?" |
| **F1 score** | A single number combining precision and recall. Ranges 0 to 1, higher is better. |
| **Macro F1** | The average F1 across all seven emotions, treating each emotion as equally important regardless of how many examples it has. This is the fairest single number here. |
| **Balanced accuracy** | The average recall across all seven emotions. Tells you how the model does if every emotion mattered equally. |
| **Confusion matrix** | A grid showing what the model predicted for each true emotion. The diagonal is correct answers, everything off the diagonal is a mistake, and the pattern of mistakes tells you a lot. |
| **Class imbalance** | Some emotions have far more training photos than others. Happy has 7,215, disgust has 436. |
| **Overfitting** | The model memorises the training photos instead of learning general patterns, so it does well in practice and badly in the exam. |
| **Data augmentation** | Randomly rotating, zooming and flipping training images so the model sees more variety and memorises less. |

---

## Results at a glance

| Measure | Value | What it tells you |
|---------|-------|-------------------|
| **Test accuracy** | **66.45%** | Share of the 7,178 unseen images classified correctly |
| 95% confidence interval | 65.36% to 67.55% | The range the true figure plausibly sits in |
| Test loss | 0.9347 | Technical measure of how wrong the model's confidence was |
| **Macro F1** | **0.6302** | The fair, per-emotion-weighted score |
| Weighted F1 | 0.6597 | F1 weighted by how common each emotion is |
| Macro precision | 0.6650 | Average trustworthiness of a prediction |
| Balanced accuracy (macro recall) | 0.6183 | Average catch rate across the seven emotions |

**Reading this in plain English:** the model gets about two out of three faces right overall. But if you gave it an exam with equal numbers of every emotion, it would score about 62%, not 66%. That 4.6 point difference is the part of the headline number that comes from the test set being full of easy, plentiful happy faces rather than from real skill spread evenly across the task.

---

## The seven emotions and the imbalance problem

| Label | Emotion | Training images | Test images |
|-------|---------|-----------------|-------------|
| 0 | Angry | 3,995 | 958 |
| 1 | **Disgust (rarest)** | 436 | 111 |
| 2 | Fear | 4,097 | 1,024 |
| 3 | **Happy (most common)** | 7,215 | 1,774 |
| 4 | Neutral | 4,965 | 1,233 |
| 5 | Sad | 4,830 | 1,247 |
| 6 | Surprise | 3,171 | 831 |
| | **Total** | **28,709** | **7,178** |

Happy has **16.55 times more training images than disgust**. The test set is skewed almost identically, at 15.98 times.

This matters more than it sounds. A model can quietly ignore disgust entirely and barely dent its accuracy score, because disgust is only 1.5% of the test set. That is why this project reports macro F1 and balanced accuracy alongside accuracy, and why the evaluation digs into each emotion separately.

These counts were verified by the notebook itself, which walked every folder and opened every file. It found **zero corrupted images** in both the training and test directories.

---

## The dataset: FER2013

**Full name:** Facial Expression Recognition 2013
**Origin:** Created for the ICML 2013 Challenges in Representation Learning
**Size:** 35,887 images, 48x48 pixels, grayscale
**Kaggle path:** `/kaggle/input/fer2013/`

The folders look like this:

```
fer2013/
├── train/     28,709 images
│   ├── angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
└── test/       7,178 images
    ├── angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
```

### How the data was split

| Pile | Images | Share of total | Where it came from |
|------|--------|----------------|--------------------|
| Training | 22,968 | 64.0% | 80% of the `/train` folder |
| Validation | 5,741 | 16.0% | 20% of the `/train` folder |
| Test | 7,178 | 20.0% | The entire `/test` folder, untouched |

The 80/20 split is applied inside each emotion folder separately, so the mix of emotions stays the same in both piles.

The important part: **validation and test images come from physically different folders.** They cannot overlap by accident. The notebook still checks this programmatically and prints the result, which was **0 files in common**, exactly as expected.

### Known problems with this dataset

Be aware of these before drawing conclusions from any FER2013 result:

- **Low resolution.** At 48x48 pixels, some expressions are genuinely indistinguishable. This caps what any model can achieve.
- **Wrong labels.** A commonly cited estimate in the literature is that roughly 3 to 5% of images are mislabelled. Part of the model's error is therefore impossible to fix.
- **Severe imbalance**, as described above.
- **Some images are not faces at all.** The dataset was assembled by web scraping.
- **Human performance is only about 65% plus or minus 5%** (Goodfellow et al., 2013), which is the realistic ceiling.
- **No demographic labels whatsoever.** There is no record of age, gender or ethnicity, so it is impossible to check whether the model works equally well for different groups. This is a real gap, not a minor one.

---

## The model

A three-block VGG-style CNN, built from scratch. No pretrained weights.

```
Input (48, 48, 1)
│
├── Block 1
│   Conv2D(64, 3x3, padding=same, ReLU)
│   BatchNormalization
│   Conv2D(64, 3x3, padding=same, ReLU)
│   BatchNormalization
│   MaxPooling2D(2x2)      ->  (24, 24, 64)
│   Dropout(0.25)
│
├── Block 2
│   Conv2D(128, 3x3, padding=same, ReLU)
│   BatchNormalization
│   Conv2D(128, 3x3, padding=same, ReLU)
│   BatchNormalization
│   MaxPooling2D(2x2)      ->  (12, 12, 128)
│   Dropout(0.25)
│
├── Block 3
│   Conv2D(256, 3x3, padding=same, ReLU)
│   BatchNormalization
│   Conv2D(256, 3x3, padding=same, ReLU)
│   BatchNormalization
│   MaxPooling2D(2x2)      ->  (6, 6, 256)
│   Dropout(0.25)
│
└── Classifier head
    Flatten
    Dense(256, ReLU)
    BatchNormalization
    Dropout(0.5)
    Dense(7, Softmax)      ->  7 emotion probabilities
```

**Total parameters: 3,510,215** (3,507,911 trainable, 2,304 fixed batch normalisation statistics). That is about 13.4 MB, which fits comfortably on a free Kaggle T4 GPU.

### Why it is built this way

- **Filters double each block (64, then 128, then 256).** Early layers spot simple things like edges. Later layers combine those into textures, then into whole facial arrangements. As the image shrinks through pooling, the number of pattern detectors grows to compensate.
- **`padding='same'` keeps the image size steady inside each block.** Only the pooling layers shrink it. This gives the network two learning layers at each scale instead of one.
- **Batch normalisation after every convolution.** This keeps the numbers flowing through the network in a sensible range, which makes training from scratch far more stable and faster.
- **Dropout randomly switches off some connections during training.** It is set to 0.25 in the convolution blocks and a heavier 0.5 just before the final layer, because that final dense layer has the most parameters and therefore the most opportunity to memorise.
- **A single Dense(256) layer before the output.** Enough capacity to combine features into emotion-level concepts without adding millions of parameters.

---

## Training setup

| Setting | Value | Why |
|---------|-------|-----|
| Input shape | 48 x 48 x 1 (grayscale) | The dataset's native format. Upscaling adds compute, not information. |
| Batch size | 64 (359 steps per epoch) | Standard choice for a model this size on a T4 |
| Max epochs | 55 | Generous upper limit |
| Optimiser | Adam, starting learning rate 0.001 | Reliable default for image classification |
| Loss function | categorical_crossentropy | Standard for multi-class problems |
| Validation split | 0.2, applied within each emotion folder | Keeps the emotion mix identical in both piles |
| EarlyStopping | patience 8 on validation loss, restore best weights | Stops if validation loss has not improved for 8 epochs, then rewinds to the best point |
| ReduceLROnPlateau | patience 4, factor 0.2 | Cuts the learning rate to one fifth if validation loss stalls for 4 epochs |
| Random seed | 42, set before the model is built | For reproducibility |
| Hardware | Kaggle NVIDIA Tesla T4, free tier | |

### Data augmentation

Applied **only to the training images**, never to validation or test:

- `rotation_range=20`: rotate by up to 20 degrees either way
- `zoom_range=0.2`: zoom in or out by up to 20%
- `horizontal_flip=True`: mirror left to right, which is realistic because faces are roughly symmetric
- `rescale=1./255`: convert pixel values from 0 to 255 down to 0 to 1, which neural networks handle better

Deliberately not used: vertical flipping (nobody is upside down in deployment), colour jitter (the images are grayscale), and brightness shifts (FER2013 already varies a lot in brightness).

Keeping validation and test clean matters. If you augmented them too, your scores would describe performance on distorted images rather than real ones.

---

## What the notebook does, cell by cell

The notebook runs top to bottom in ten sections. If you are reading the code alongside this, the section headings match the markdown cells.

### 1. Import libraries and set the seed

Loads TensorFlow, NumPy, Matplotlib, Seaborn and PIL, and calls `tf.keras.utils.set_random_seed(42)` before anything else, so weight initialisation and shuffling are repeatable.

### 2. Data quality check

Walks every emotion folder, opens each image with PIL and calls `.verify()` to catch corruption, then prints the totals and per-class counts. This runs before training so a bad file cannot crash a 45 minute run halfway through. Result: **0 corrupted images** in both folders.

### 3. First data generators

Sets up the image loaders at 48x48, grayscale, batch size 64. Note that this first version uses a 10% validation split (25,841 training and 2,868 validation images). It is superseded in section 5.

### 4. Sample visualisation

Pulls one batch and shows 10 images with their labels in a 2 by 5 grid.

This is a small step that pays for itself. You see exactly what the model sees, and you immediately notice how hard the task is at this resolution. Fear and surprise look alike. Sad and neutral blur together. If a label were obviously wrong, you would spot it here.

### 5. Augmentation and the final generators

Rebuilds the loaders with rotation, zoom and horizontal flip turned on, and switches the validation split to 0.2. **This is the configuration actually used for training:** 22,968 training, 5,741 validation, 7,178 test.

A chart cell then draws a pie chart and a stacked bar showing the three-way split, with an annotation making clear that validation comes from the `/train` folder while test comes from the separate `/test` folder. Saved as `data_split_visualisation.png`.

### 6. Split verification

Prints the composition of the validation and test sets, emotion by emotion, then compares the two file lists and counts how many paths appear in both. The answer must be zero, and it is. Without this check, a leak between validation and test would make every score meaningless and you would never know.

### 7. Callbacks

Defines EarlyStopping (patience 8) and ReduceLROnPlateau (patience 4, factor 0.2), both watching validation loss.

Validation loss is watched rather than validation accuracy because loss changes smoothly while accuracy jumps around, especially on imbalanced data. A smoother signal makes for better stopping decisions.

### 8. Build, compile and train

Builds the CNN described above, prints the layer-by-layer summary, compiles with Adam and categorical cross-entropy, then runs `model.fit()` for up to 55 epochs.

A hyperparameter audit cell follows, printing every single setting used. This means anyone reading the published notebook can see the full configuration without reverse-engineering it from the code.

### 9. Training performance charts

Draws a three-panel figure, saved as `training_performance.png`:

1. Training and validation **accuracy** across epochs
2. Training and validation **loss** across epochs
3. The **gap** between training and validation accuracy, with a zero line for reference

Panel 3 is the overfitting detector. A gap that stays small and flat is healthy. A gap that widens steadily means the model is memorising.

### 10. Evaluation

Three things happen here:

- A **four-panel evaluation figure** (`comprehensive_evaluation.png`): raw confusion matrix, percentage confusion matrix, per-emotion F1 bars with the macro average marked, and an overall metrics summary
- A **full classification report** with precision, recall and F1 for every emotion
- **`model.evaluate()`** on the test set for the final headline figures

### 11. Qualitative check on 200 random images

Picks 200 test images at random without replacement, seeded at 42, and draws them in a 10 by 20 grid. Each image is titled with its true label, the model's prediction and the confidence percentage. Green title for correct, red for wrong. Saved as `200_test_predictions.png`.

**Result: 129 of 200 correct, which is 64.5%.**

That is slightly below the 66.45% full-test figure, and it is worth checking whether the difference means anything. It does not. Drawing 200 images from a pool of 7,178 containing 4,770 correct predictions gives an expected count of 132.9 with a standard deviation of 6.58, so 129 sits **0.59 standard errors below expectation**. That is comfortably within normal sampling variation, so this grid is a representative sample and you can trust your visual impression of it.

This is worth doing every time. A 200 image sample carries a 95% confidence interval of roughly 13 points, so an unlucky draw can easily look like a broken model. The panels of a dozen images common in papers are far wider still, and without a check like this the reader has no way to know.

---

## What actually happened during training

Training ran the **full 55 epochs**. Early stopping fired on the last epoch, having gone 8 epochs without improvement, and **restored the weights from epoch 47**, which had the lowest validation loss of 0.9418.

The learning rate was cut five times as validation loss plateaued:

| Epoch | New learning rate | Effect |
|-------|-------------------|--------|
| 25 | 0.0002 | The clearest single improvement of the run. Validation accuracy rose from 62.15% to 64.19% in one epoch and validation loss fell from 1.0256 to 0.9701. The model had been oscillating rather than converging at the original rate. |
| 33 | 0.00004 | Small further gain |
| 46 | 0.000008 | Produced the best epoch, 47 |
| 51 | 0.0000016 | No meaningful gain |
| 55 | 0.00000032 | No meaningful gain, training stopped |

### Was it overfitting?

At the selected epoch 47, training accuracy was about **72.08%** against validation accuracy **66.28%**, a gap of roughly **5.8 points**.

That is mild and controlled. Some gap is always expected, because the model has seen the training images many times. A gap that stays stable rather than growing means dropout, batch normalisation and augmentation were doing their jobs, and early stopping caught the run before it deteriorated.

Note: per-epoch training accuracy figures come from the Keras progress log, which reports a running average across the epoch, so they can differ by a few tenths of a point from the stored history values.

---

## Results in detail

Everything below was read from the stored outputs of this notebook, or worked out arithmetically from the classification report and the class counts. Nothing is estimated.

### Per-emotion performance

| Emotion | Precision | Recall | F1 | Test images |
|---------|-----------|--------|-----|-------------|
| Angry | 0.556 | 0.622 | 0.587 | 958 |
| Disgust | 0.774 | 0.432 | 0.555 | 111 |
| Fear | 0.563 | **0.368** | **0.445** | 1,024 |
| Happy | 0.880 | 0.870 | **0.875** | 1,774 |
| Neutral | 0.569 | 0.724 | 0.637 | 1,233 |
| Sad | 0.557 | 0.535 | 0.546 | 1,247 |
| Surprise | 0.756 | 0.776 | 0.766 | 831 |
| **Macro average** | **0.665** | **0.618** | **0.630** | 7,178 |
| Weighted average | 0.666 | 0.665 | 0.660 | 7,178 |

**The spread is the real finding.** Happy reaches F1 0.875 and surprise 0.766, both well above the overall figure. Fear sits at 0.445 and disgust at 0.555. Fear recall of 0.368 means the model **misses nearly two thirds of fearful faces**. Reporting only the 66.45% headline would hide that completely.

### How often each emotion was predicted

These counts follow exactly from the precision, recall and support figures above, and they sum to 7,178 as they must.

| Emotion | Times predicted | Times it truly occurred | Ratio | Reading |
|---------|-----------------|-------------------------|-------|---------|
| Disgust | 62 | 111 | 0.56 | Heavily under-predicted |
| Fear | 670 | 1,024 | 0.65 | Heavily under-predicted |
| Sad | 1,197 | 1,247 | 0.96 | Balanced |
| Happy | 1,755 | 1,774 | 0.99 | Balanced |
| Surprise | 853 | 831 | 1.03 | Balanced |
| Angry | 1,072 | 958 | 1.12 | Over-predicted |
| Neutral | 1,569 | 1,233 | 1.27 | Over-predicted |

The two over-predicted labels are absorbing the missing mass. Neutral picks up 676 false positives and angry 530, which is where the missed fear and disgust faces end up going. The exact destination of every error is shown in the confusion matrix panel of `comprehensive_evaluation.png`.

### Two different failure modes, not one

It is tempting to blame all of this on disgust simply having too few examples. The numbers do not support such a simple story.

The rank correlation between how many training images an emotion has and how well the model recalls it is **rho = 0.43 (p = 0.34, n = 7)**. That is a weak, statistically insignificant association, and two emotions contradict it outright:

- **Surprise** is the second rarest emotion but achieves the second highest recall
- **Fear** is the third most common but has the lowest recall of all

Comparing precision against recall separates the two mechanisms:

**Disgust looks like scarcity.** The model uses the label only 62 times where 111 images carry it, but when it does commit, it is right about 77% of the time. High precision with low recall is the classic fingerprint of an under-trained class: with so few examples contributing to the loss, the model has drawn a cautious boundary and only says "disgust" when the evidence is overwhelming. This is exactly the failure that class weighting or focal loss is designed to fix.

**Fear looks like confusability.** Fear is under-predicted too, but its precision is also poor at 0.563, so the errors run in both directions. The model both misses fearful faces and wrongly calls other faces fearful. With 4,097 training images available, scarcity cannot be the explanation. This looks like genuine visual ambiguity at 48x48 pixels, where a wide-eyed fearful expression and a wide-eyed surprised one are only a few pixels apart.

**Why this distinction matters practically:** one fix will not address both. Rebalancing the classes should help disgust while leaving fear roughly where it is. Fear needs either higher resolution inputs or an architecture that separates confusable expressions better, such as attention modules. Fear therefore makes a useful control variable: if you add class weighting and fear improves as much as disgust, this two-mechanism reading is wrong.

---

## Why accuracy on its own would mislead you

Here is a concrete demonstration, using only public properties of the test set.

Overall accuracy is the support-weighted average of per-class recall. So the highest accuracy you could reach while completely failing on a group of classes is one minus their combined share of the test set.

Take the three smallest classes: disgust (111), surprise (831) and angry (958). Together they are 26.47% of the test set. **A model that recognised no anger, no disgust and no surprise at all could still score 73.53%.**

That figure is higher than the 73.28% published for a carefully tuned VGGNet on this benchmark. In a league table sorted by accuracy, a model blind to three of the seven emotions would outrank a genuine state of the art system.

This is why the repository leads with macro F1 and balanced accuracy.

### The cost of aggregation, measured

| Quantity | Value |
|----------|-------|
| Accuracy minus balanced accuracy | **4.62 points** |
| Accuracy minus macro F1 | 3.43 points |
| Weighted F1 minus macro F1 | 0.030 |

The 4.62 point gap is larger than the entire 95% confidence interval around the accuracy estimate, which is about 2.2 points wide. In other words, the distortion from ignoring class balance is bigger than the measurement uncertainty. That is the clearest argument for reporting both.

### A quick check that the test set stayed clean

| Quantity | Value |
|----------|-------|
| Validation accuracy at epoch 47 | 66.28% |
| Test accuracy | 66.45% |
| **Test minus validation** | **+0.17 points** |

The sign here is what matters. If model selection had leaked into the test set, you would expect validation to look **better** than test, because the checkpoint was chosen using validation data. The opposite is observed, by a small margin. The validation estimate was very slightly conservative, which is what a clean protocol looks like.

---

## How this compares to published work

| System | Approach | Accuracy |
|--------|----------|----------|
| Random guessing | Property of a 7-class problem | 14.29% |
| Always guess "happy" | Property of the test distribution | 24.71% |
| **This model, balanced accuracy** | *class-balanced view* | **61.83%** |
| Human annotators (Goodfellow et al., 2013) | Human baseline | 65% plus or minus 5% |
| **This model, accuracy** | 3-block CNN trained from scratch | **66.45%** |
| Tang (2013) | CNN with an L2-SVM objective | 71.16% |
| Khaireddin and Chen (2021) | Heavily tuned VGGNet | 73.28% |
| *"Blind to 3 classes" ceiling* | *Property of the test distribution* | *73.53%* |
| Pramerdorfer and Kampel (2016) | Ensemble of CNNs | 75.2% |

This model sits below the tuned and ensembled systems, which is the honest position for a single from-scratch network of this size with no pretraining. The published figures above are reported as stated in those papers and have not been independently re-run here.

The 66.45% result falls inside the 65 plus or minus 5% band reported for human annotators, which is the right way to state it. Claiming "human parity" from a single point estimate would be overclaiming.

---

## What the evidence does not support

Worth stating openly rather than leaving buried.

- **Training set size does not fully explain which emotions fail.** The correlation is rho = 0.43 with p = 0.34, and with seven data points this is a weak test. The two-mechanism account is an interpretation supported by the precision and recall pattern, not a proven causal claim.
- **The fairness discussed here is between emotion classes, not between people.** FER2013 carries no age, gender or ethnicity labels, so the disparity measures the fairness literature asks for cannot be computed at all. A model that treats all seven emotions evenly could still treat demographic groups very unevenly, and nothing in this notebook would detect it.
- **This is one training run.** A fixed seed makes the run repeatable, but it does not make it typical. The confidence interval quoted covers test set sampling only, not run-to-run variation. Three to five seeds with a mean and standard deviation would be a much stronger claim.
- **Part of the error is unfixable.** FER2013's known label noise sets a ceiling that no architecture can cross on this dataset. Confirming any improvement properly means testing on a relabelled or in-the-wild successor such as FER+ or AffectNet.
- **The confusion matrix detail lives in the figure, not in this file.** Per-pair error percentages are visible in `comprehensive_evaluation.png`, and the derived prediction counts above were computed from the classification report rather than re-read from the matrix.

---

## How to reproduce

### Option 1: Kaggle, recommended

1. Sign in at [Kaggle](https://www.kaggle.com/) and create a new notebook
2. In the data tab, search for `msambare/fer2013` and click **Add**
3. Upload `facial-expression-recognition-with-cnn.ipynb`, or copy the cells across
4. In notebook settings, turn on **GPU T4 x1** (free tier)
5. Run all cells top to bottom

**Expected runtime: roughly 45 to 55 minutes** on a T4. Epochs took between 40 and 56 seconds each across the 55 epoch run.

The seed is fixed at 42 before the model is built, and the train/validation split slices each emotion folder's sorted file list by index, so the same code on the same data produces the same partition. Exact reproduction of the final decimal place is not guaranteed, because GPU floating point operations are not bitwise deterministic by default.

### Option 2: Locally

```bash
git clone https://github.com/CollinsLemeke/Facial-Expression-Recognition-Model.git
cd Facial-Expression-Recognition-Model

pip install -r requirements.txt

# You need a Kaggle API token: https://www.kaggle.com/docs/api
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/

# Change the paths in the notebook from /kaggle/input/fer2013 to data/fer2013
jupyter notebook facial-expression-recognition-with-cnn.ipynb
```

### Hardware guidance

- **CPU only:** works, but expect 6 to 10 hours. Not recommended.
- **Any modern GPU** (T4, RTX 3060 or better, A10G): 45 to 55 minutes
- **A100 or L4:** 15 to 25 minutes

---

## Repository structure

```
.
├── README.md
├── facial-expression-recognition-with-cnn.ipynb
├── requirements.txt
├── outputs/                                  (generated when you run the notebook)
│   ├── data_split_visualisation.png
│   ├── training_performance.png
│   ├── comprehensive_evaluation.png
│   └── 200_test_predictions.png
└── LICENSE
```

The notebook is self-contained. Running it end to end produces all four figures, which are the ones you would put in a paper or dissertation appendix.

Note that the notebook does not currently save the trained model weights. If you want to reuse the model without retraining, add `model.save('fer_cnn.keras')` after training.

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

```bash
pip install -r requirements.txt
```

On Kaggle all of these are pre-installed, so no setup is needed.

---

## Limitations and ethics

Facial expression recognition is a sensitive area. These points should be read before any use beyond research.

**Dataset limitations**

- 48x48 resolution caps achievable accuracy
- Severe class imbalance, 16.55 to 1 between happy and disgust in training
- Roughly 3 to 5% of labels are estimated to be wrong
- No demographic annotation at all, so no fairness audit is possible on this data
- Images were web-scraped with limited consent metadata

**Ethical considerations**

- **A facial expression is not an emotion.** This model predicts the label a human annotator would assign to a facial configuration. It does not detect what a person feels. Context, culture and individual differences make that a probabilistic inference at best. With balanced accuracy of 61.83% and fewer than four in ten fearful faces recognised, this is not a reliable instrument for any decision that affects someone.
- **Cultural bias.** FER datasets skew Western in both who was photographed and how the labels were assigned. Deploying on a different population without local validation is not defensible.
- **Regulatory position in the EU.** Article 5(1)(f) of Regulation (EU) 2024/1689 (the EU AI Act) prohibits placing on the market or using AI systems to infer emotions of a person in workplace and education settings, other than for medical or safety reasons. That prohibition has applied since 2 February 2025. Two applications often used to motivate FER research, classroom engagement monitoring and workplace affect analytics, therefore fall inside a prohibited category. This model must not be used for either.
- **Fairness auditing is a prerequisite, not an extra.** Before any real deployment a model like this needs testing for performance gaps across demographic groups. FER2013 makes that impossible, which is itself a reason to move to a better annotated dataset.

This notebook is intended for research, education and academic benchmarking. It is not production ready for any consequential use.

---

## Roadmap

Ordered by what the results above actually justify.

1. **Save the trained model.** A one-line change that makes the run reusable.
2. **Multi-seed evaluation.** Run three to five seeds and report mean and standard deviation instead of a single point estimate. This is the highest value next step.
3. **Class weighting or focal loss.** Compare against the per-class baseline recorded here, using fear as a control that should *not* improve much if the two-mechanism reading is correct. Note that class weighting has not actually been tested in this notebook yet.
4. **Attention modules (SE or CBAM) or higher resolution inputs.** Aimed at the confusability problem rather than the scarcity one.
5. **Cross-dataset validation on AffectNet or RAF-DB.** These carry demographic annotation, which opens the fairness question FER2013 closes off.
6. **Transfer learning baselines.** VGG16, ResNet50 or EfficientNet-B0 pretrained on ImageNet.
7. **Test-time augmentation.** Average predictions across several augmented views of each test image.
8. **Model ensembling.** Combine three to five independently trained models.
9. **Face detection preprocessing.** MTCNN or MediaPipe to crop faces more tightly first.
10. **Grad-CAM visualisations.** Show which parts of the face drive each prediction, keeping in mind that saliency shows where the evidence is, not why the model decided.
11. **FER+ support.** FER2013's better-labelled successor.
12. **Real-time webcam demo and knowledge distillation** into a smaller mobile-sized model.

---

## Which run these numbers come from

Every figure in this README was taken from the stored outputs of the notebook in this repository, or derived arithmetically from them. If you retrain, the numbers will shift slightly and this file will need updating.

Quick reference for the run recorded here:

- 55 epochs completed, best weights restored from epoch 47
- Validation loss at the best epoch: 0.9418
- Test accuracy 66.45%, test loss 0.9347, macro F1 0.6302
- 200-image qualitative grid: 129 of 200 correct

---

## Author

**Collins Lemeke**

AI research and engineering. Research work with the Centre of Intelligence of Things (CIoTh), University of Greater Manchester.

This project sits within a wider interest in efficient, accessible computer vision and affective computing, alongside work on lightweight NLP for mental health sentiment analysis and carbon-aware model design.

- [Kaggle notebook](https://www.kaggle.com/code/collinslemeke/facial-expression-recognition-with-cnn)
- [GitHub](https://github.com/CollinsLemeke)

For questions, feedback or suggestions, open a GitHub issue.

---

## License

MIT License. Free to use, modify and distribute. See [LICENSE](LICENSE) for the full terms.

The FER2013 dataset has its own licence and terms of use, separate from this code. See the [original Kaggle dataset page](https://www.kaggle.com/datasets/msambare/fer2013) for details.
