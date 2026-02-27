# 🫀 Heart Disease Prediction – Kaggle Playground Series S6E2

**Author:** Arasirwa William
**Evaluation Metric:** ROC-AUC
**Best Public LB Score:** `0.95372`

## 📌 Project Overview

This repository contains a production-ready machine learning pipeline developed for the Kaggle Playground Series (Season 6, Episode 2). The objective is to predict the probability of a patient having heart disease based on clinical metrics.

This solution overcomes the challenges of noisy synthetic medical data through rigorous exploratory data analysis, domain-specific feature engineering, and a heavily regularized, 7-model stacking ensemble.

---

## 🔬 1. Exploratory Data Analysis & "Digit Preference"

The dataset contained synthetic artifacts amplified from real-world medical data. During EDA, a phenomenon known as **"Digit Preference"** was identified:

* **Blood Pressure & Cholesterol:** Massive, unnatural spikes were observed at exact multiples of 10 (e.g., BP at 120, 130, 140; Cholesterol at 200, 240). This represents the synthetic AI mimicking human doctors rounding measurements.
* **Outliers:** Extreme values (e.g., BP > 180, Cholesterol > 400) were retained, as they represent biologically valid, high-risk clinical indicators rather than erroneous data.

---

## 🧬 2. Clinical Feature Engineering

Tree-based algorithms excel at finding thresholds but struggle with raw arithmetic. To assist the models, medical domain knowledge was injected directly into the dataset:

* **`hr_efficiency`:** Calculated the patient's theoretical maximum heart rate (`220 - age`) and created a ratio against their actual achieved `max_hr`. This provides a direct mathematical measure of cardiovascular strain.
* **`bp_category`:** Grouped the spiky, digit-preferenced Blood Pressure data into smooth, recognized clinical buckets (Normal, Elevated, Stage 1, Stage 2) to prevent algorithms from memorizing synthetic noise.

---

## ⚙️ 3. Validation Strategy & Baseline

To ensure the models generalized perfectly to unseen data and avoided the "Public Leaderboard Trap," the entire architecture was wrapped in a **5-Fold Stratified Cross-Validation** (`StratifiedKFold`).

A Logistic Regression baseline was established, achieving an Out-Of-Fold (OOF) ROC-AUC of `0.9530`. Learning curve analysis confirmed the baseline was perfectly fitted (zero variance between training and validation), proving the engineered features successfully captured the underlying clinical patterns.

---

## 🧠 4. The Final Architecture: Ridge-Stacked Fusion

The final `0.95372` pipeline utilizes an elite Level 1 Stacking Ensemble designed specifically to defeat multicollinearity.

### 1. Seed Ensembling (7 Base Models)

To maximize variance reduction, 7 highly regularized gradient boosting models were deployed:

* 2 × CatBoost
* 2 × LightGBM
* 3 × XGBoost

Each model used identical, heavily penalized hyperparameters (e.g., `l2_leaf_reg = 10`, `max_depth = 3`) but different `random_seed` values. This forced the tree algorithms to build slightly different architectures, feeding the stacker a richer diversity of predictions.

### 2. The Meta-Model (RidgeClassifier)

Because Gradient Boosters output highly correlated predictions, standard meta-models (like Logistic Regression) suffer from multicollinearity and assign destructive negative weights.
This pipeline uses a `RidgeClassifier`. The native L2 Regularization in Ridge perfectly handles collinearity, distributing weights evenly across all 7 models to find the ultimate consensus.

### 3. The Sigmoid Hack

Because Ridge outputs raw distance metrics instead of probabilities, a Sigmoid transformation was applied to the final predictions:
$P = \frac{1}{1 + \exp(-x)}$
This elegantly squashed the raw distances back into the perfect 0-to-1 scale required for Kaggle's ROC-AUC evaluation.