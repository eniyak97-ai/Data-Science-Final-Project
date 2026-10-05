# Breast Cancer Diagnosis Prediction

An end-to-end data science and machine learning project that predicts whether a breast tumour is **malignant** or **benign** from measurements of cell nuclei.
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Project Workflow](#project-workflow)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Modelling](#modelling)
7. [Results](#results)
8. [Key Insights](#key-insights)
9. [Limitations](#limitations)
10. [Future Work](#future-work)
11. [How to Run](#how-to-run)
12. [Repository Structure](#repository-structure)
13. [Tech Stack](#tech-stack)
14. [Author](#author)

---

## Project Overview
This project walks through the full data science lifecycle: understanding the data, exploring it visually, building and comparing several machine learning models, tuning the best one, evaluating it on unseen data, and interpreting what drives its predictions. It was completed as the final project of my data science course and is designed to be portfolio-ready and fully reproducible.

## Problem Statement
Early and accurate diagnosis of breast cancer improves patient outcomes. Given numerical measurements computed from digitised images of a fine-needle aspirate of a breast mass, can we build a model that reliably classifies a tumour as malignant or benign, and tell us which measurements matter most?

## Dataset
- **Name:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Source:** Built into scikit-learn (`sklearn.datasets.load_breast_cancer`), so no download is needed
- **Size:** 569 samples, 30 numeric features
- **Target:** Diagnosis (1 = malignant, 0 = benign)
- **Class balance:** about 63% benign and 37% malignant
- **Data quality:** no missing values and no duplicate rows

The 30 features are the **mean**, **standard error** and **worst (largest)** values of 10 characteristics of the cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

## Project Workflow
1. **Data understanding:** shape, data types, missing values, duplicates, class balance
2. **Exploratory data analysis:** distributions, group comparisons, correlations
3. **Train/test split:** 80/20, stratified to preserve class proportions
4. **Model comparison:** 5-fold stratified cross-validation on the training set
5. **Hyperparameter tuning:** GridSearchCV on the strongest candidate
6. **Final evaluation:** confusion matrix, ROC curve and classification report on the held-out test set
7. **Interpretation:** permutation importance and error analysis
8. **Conclusions:** insights, limitations and next steps

## Exploratory Data Analysis
- The classes are moderately imbalanced, so I used stratified splits and focused on recall and F1 instead of accuracy alone.
- Size and shape features (radius, perimeter, area, concavity, concave points) show the strongest relationship with malignancy.
- Radius, perimeter and area are almost perfectly correlated, which means there is heavy multicollinearity in the data.

## Modelling
Four models were compared using 5-fold stratified cross-validation. Scaling was placed inside a scikit-learn `Pipeline` so that no information leaks from validation folds into training.

| Model | CV Recall | CV F1 | CV ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.953 | 0.964 | 0.996 |
| SVM (RBF) | 0.947 | 0.961 | 0.995 |
| Gradient Boosting | 0.947 | 0.955 | 0.991 |
| Random Forest | 0.935 | 0.946 | 0.988 |

The SVM was then tuned with GridSearchCV over `C` and `gamma`. The best settings were `C=10` and `gamma='scale'`, with a cross-validated F1 of about 0.967.

## Results
Final performance of the tuned SVM on the **held-out test set** (114 samples):

| Metric | Score |
|---|---|
| Accuracy | 97.4% |
| ROC-AUC | 0.993 |
| Malignant precision | 100% |
| Malignant recall | 92.9% |
| Benign precision | 96.0% |
| Benign recall | 100% |

**Confusion matrix summary:** all 72 benign cases were classified correctly. Of the 42 malignant cases, 39 were caught and 3 were missed (false negatives). There were no false positives.

## Key Insights
- **The "worst" measurements are the most informative.** Worst concave points, worst radius, worst perimeter and worst area are the strongest predictors (permutation importance).
- **Simple models compete with complex ones.** Logistic Regression had the best cross-validation F1, which suggests the classes are close to linearly separable. A transparent model is a sound choice here.
- **Recall matters most in a medical setting.** A missed cancer (false negative) is far more costly than a false alarm, so the decision threshold should be tuned for higher recall rather than left at 0.5.

## Limitations
- The dataset is small (569 rows) and comes from a single source, so performance on other hospitals or imaging setups is unproven.
- The model has not been externally validated.
- Predicted probabilities have not been calibrated.
- This is an educational project and **must not be used for real medical decisions**.

## Future Work
- Tune the probability threshold to reach a target recall, and calibrate the probabilities
- Apply feature selection or PCA to reduce correlated features
- Add SHAP explanations for individual predictions
- Validate on an external dataset
- Deploy the model as an interactive Streamlit web app

## Tech Stack
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn
