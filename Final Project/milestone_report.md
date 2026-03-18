# Milestone Report: Classifying American Sign Language Alphabet Images Using Machine Learning

**Course:** DS4400 Data Mining and Machine Learning 1

**Professor:** Silvio Amir

**TA:** [TA Name]

**Team Members:** Dev Patel, Hareg Aderie

**University:** Northeastern University

---

## Problem Description

We are solving a **multi-class image classification** problem: predicting the correct American Sign Language (ASL) alphabet letter from an image of a hand sign. Given a 200x200 RGB image of a hand forming an ASL letter, the model must classify it into one of 29 categories (A-Z, SPACE, DELETE, NOTHING).

This is a **classification problem** with 29 classes.

### Motivation

Accurate ASL recognition can support accessibility tools such as sign translation systems, educational applications, and human-computer interaction technologies for people with hearing loss. The ASL alphabet provides a clear supervised learning task for comparing classical machine learning and deep learning approaches in computer vision.

### Related Work

- **Pugeault and Bowden (2011):** One of the earlier works on ASL finger-spelling recognition using depth sensors and random forests, establishing baselines for hand shape classification.
- **Kaggle community notebooks** on the ASL Alphabet dataset demonstrate that CNNs can achieve over 95% accuracy on this dataset, with common architectures including VGG-style and ResNet-based models.
- **OpenCV and MediaPipe hand tracking tutorials** show how real-time hand landmark detection can be combined with classifiers for live ASL recognition, which informs our future work on a live tracker.
- **Transfer learning surveys (Zhuang et al., 2020)** show that pretrained models like MobileNet and ResNet can be effectively fine-tuned for domain-specific image classification tasks, even with relatively small datasets.

---

## Dataset

### Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Source | Kaggle ASL Alphabet Database |
| Total images | ~87,000 |
| Number of classes | 29 (A-Z + SPACE, DELETE, NOTHING) |
| Images per class | ~3,000 |
| Image resolution | 200 x 200 pixels |
| Color channels | 3 (RGB) |
| Features per image (raw) | 200 x 200 x 3 = 120,000 |
| Missing data | None |
| Class balance | Balanced (~3,000 per class) |

### Feature Definition and Insights

Each image is represented as a 200x200x3 array of pixel intensities (0-255). For classical models, we extract HOG (Histogram of Oriented Gradients) features, which reduce dimensionality while preserving edge and shape information. For deep learning models, we normalize pixel values to [0, 1] and feed raw RGB images.

Key observations from exploring the data:
- Images contain a single hand against varied backgrounds
- Lighting conditions and skin tones vary across images
- Some letters have very similar hand shapes (M/N, R/U, V/W), which may cause confusion
- The NOTHING class contains images with no hand, serving as a negative class

### Exploratory Data Analysis

We performed the following EDA:
- **Class distribution:** Confirmed balanced classes (~3,000 images each)
- **Sample image visualization:** Displayed representative images per class to understand visual differences
- **Pixel intensity histograms:** Analyzed RGB channel distributions to inform normalization
- **HOG feature visualization:** Compared raw images with HOG representations to verify edge capture

---

## Approach and Methodology

### General Approach

1. **Preprocessing:** Resize images to 64x64, normalize to [0,1], extract HOG features for classical models
2. **Data split:** 70% training, 15% validation, 15% test (stratified)
3. **Data augmentation:** Rotation, shifts, and zoom for CNN training (no horizontal flip since ASL signs are hand-specific)
4. **Model training:** Train 5 models and compare performance
5. **Evaluation:** Accuracy, macro-averaged precision, recall, F1 score, and confusion matrices

### Feature Engineering

- **HOG features:** 9 orientations, 8x8 pixels per cell, 2x2 cells per block. This reduces each image from 12,288 raw features to a compact feature vector capturing gradient orientations.
- **Normalization:** Pixel values scaled to [0, 1] for neural network stability.

### Machine Learning Models

We have trained the following models with preliminary results:

#### Model 1: Multinomial Logistic Regression (Baseline)
- **Features:** HOG
- **Configuration:** L-BFGS solver, multinomial, max_iter=5000
- **Status:** Trained and evaluated

#### Model 2: SVM with RBF Kernel
- **Features:** HOG
- **Configuration:** C=10, gamma='scale'
- **Status:** Trained and evaluated

#### Model 3: Random Forest
- **Features:** HOG
- **Configuration:** 200 estimators, max_depth=30
- **Status:** Trained and evaluated

#### Model 4: CNN (Custom Architecture)
- **Features:** Raw 64x64 RGB images
- **Architecture:** 3 Conv2D+BatchNorm+MaxPool blocks, Dense(256)+Dropout(0.5), Softmax(29)
- **Training:** Adam optimizer, data augmentation, early stopping
- **Status:** Trained and evaluated

#### Model 5: Transfer Learning (MobileNetV2)
- **Features:** Raw 64x64 RGB images
- **Architecture:** Frozen MobileNetV2 base, GlobalAveragePooling2D, Dense(256)+Dropout(0.5), Softmax(29)
- **Status:** Trained and evaluated

### Preliminary Results

| Model | Test Accuracy | Test Precision (macro) | Test Recall (macro) | Test F1 (macro) |
|-------|--------------|----------------------|--------------------|-----------------|
| Logistic Regression | TBD | TBD | TBD | TBD |
| SVM (RBF) | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| CNN | TBD | TBD | TBD | TBD |
| MobileNetV2 (TL) | TBD | TBD | TBD | TBD |

*(Values will be filled after running the notebook with the full dataset)*

### Challenges and Changes from Proposal

- **Memory constraints:** Loading all 87,000 images at full resolution (200x200) into memory is not feasible for classical models. We subsample to 1,000 images per class and resize to 64x64.
- **Training time:** SVM training is slow on large datasets. We use a subset for classical models while deep learning models can leverage data generators.
- **No major changes from proposal:** We are following the original plan. The only adjustment is using subsampling for classical models and data generators for deep learning models.

---

## Remaining Work

### Live ASL Tracker

The primary remaining work is building a **real-time ASL recognition system** using the best-performing model:

1. **Webcam integration:** Use OpenCV to capture live video frames from a webcam
2. **Hand detection:** Use MediaPipe Hands to detect and crop the hand region from each frame
3. **Real-time classification:** Feed the cropped hand image through the trained CNN/MobileNetV2 model and display the predicted letter on screen
4. **Smoothing:** Implement a prediction buffer to avoid flickering between predictions (e.g., majority vote over the last 5 frames)
5. **Word building:** Accumulate predicted letters into words, using the SPACE and DELETE classes for text editing
6. **UI overlay:** Display the predicted letter, confidence score, and accumulated text on the video feed

### Additional Remaining Tasks

- Fine-tune MobileNetV2 (unfreeze top layers) for potentially higher accuracy
- Conduct detailed error analysis on commonly confused letter pairs
- Finalize the comparison report and presentation

---

## Team Member Contribution

### Hareg Aderie
**Completed:**
- Dataset download and preparation
- Exploratory data analysis (class distribution, sample visualization, pixel statistics)
- Image preprocessing pipeline (resize, normalize, HOG extraction)
- Implementation of classical models (Logistic Regression, SVM, Random Forest)
- Evaluation metrics computation and comparison

**Planned:**
- Detailed error analysis and confusion matrix interpretation
- Help with final report results section

### Dev Patel
**Completed:**
- CNN architecture design and implementation
- Data augmentation strategy
- Transfer learning implementation (MobileNetV2)
- Training curves analysis
- Model comparison framework

**Planned:**
- Live ASL tracker development (webcam + MediaPipe + model inference)
- Fine-tuning experiments for transfer learning
- Lead final report and presentation
- Real-time prediction smoothing and UI implementation

### Joint Work
- Experiment setup and dataset splitting strategy
- Model comparison and analysis
- Final review of all results and report
