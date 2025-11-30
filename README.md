# Loan Default Risk Prediction

A complete machine learning project for **binary classification**, focusing on predicting **credit card client default** risk using the Default of Credit Card Clients dataset.

The project features a full end-to-end ML workflow including data cleaning, preprocessing, class imbalance handling, cross-validation, hyperparameter tuning, and threshold optimization.

---

## Key Features

* **End-to-End ML Pipeline:** From loading raw data → EDA → preprocessing → modeling → tuning → evaluation.
* **Imbalanced Data Handling:** Applied **SMOTE** (Synthetic Minority Oversampling Technique) to oversample the minority (default) class.
* **Model Comparison:** Evaluated **Random Forest**, **XGBoost**, and **Logistic Regression** using 5-fold cross-validation.
* **Model Optimization:** Performed **Grid Search + Threshold Tuning** to improve recall and F1-score on the minority class (default clients).

---

## Dataset

This project uses the **Default of Credit Card Clients** dataset from Kaggle.

[Dataset Link](https://www.kaggle.com/datasets/mariosfish/default-of-credit-card-clients)

### Details

| Attribute | Value |
| :--- | :--- |
| **Type** | Tabular classification dataset |
| **Target Variable** | **`default.payment.next.month`** |
| **Classes** | **0:** Non-Default, **1:** Default |
| **Characteristics** | Includes features like limit balance, gender, education, marriage status, age, and past payment history (bill amounts and payments). The dataset is imbalanced. |

---

## Tech Stack

| Category | Components |
| :--- | :--- |
| **Language** | Python |
| **ML Libraries** | `scikit-learn`, `xgboost`, `imblearn` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Tools** | `GridSearchCV`, `SMOTE`, `train_test_split` |

---

## Project Pipeline

1.  ### Data Preparation & EDA
    * Cleaned and renamed columns using the data dictionary.
    * Converted categorical features (like Education and Marriage) based on domain understanding.
    * Visualized:
        * Class imbalance
        * Feature distributions
        * Correlation heatmaps

2.  ### Feature Preprocessing
    * Separated numerical and categorical features.
    * Applied:
        * **One-Hot Encoding** (for discrete categorical columns)
        * **Scaling** (for numeric columns, e.g., Standard Scaler)
    * Performed train/test split using stratification.
    * Applied **SMOTE** only on the training set to address class imbalance.

3.  ### Model Training & Cross-Validation
    Trained and compared three models using **5-fold Cross-Validation**.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | $0.866$ | $0.879$ | $0.849$ | $0.864$ |
| **XGBoost** | $0.863$ | $0.899$ | $0.817$ | $0.856$ |
| **Logistic Regression** | $0.677$ | $0.681$ | $0.666$ | $0.674$ |

4.  ### Model Evaluation
    Evaluated both Random Forest and XGBoost using:
    * Classification Report
    * Confusion Matrix
    * ROC-AUC Curve
    * Precision-Recall Curve
    * Top Feature Importances

5.  ### Hyperparameter Tuning
    Performed Grid Search for both Random Forest and XGBoost to improve model performance. After tuning, **Threshold Optimization** was performed to specifically improve the minority-class recall (detecting defaults).

---

## Final Results (After Tuning + Threshold Adjustment)

### Random Forest (Tuned Threshold)

| Class | Precision | Recall | F1-score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **0 (Non-Default)** | $0.87$ | $0.83$ | $0.85$ | $4673$ |
| **1 (Default)** | $0.49$ | $0.56$ | $0.52$ | $1327$ |

* **Accuracy:** $0.77$
* **Weighted F1:** $0.78$

### XGBoost (Tuned Threshold)

| Class | Precision | Recall | F1-score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **0 (Non-Default)** | $0.87$ | $0.82$ | $0.84$ | $4673$ |
| **1 (Default)** | $0.47$ | $0.57$ | $0.51$ | $1327$ |

* **Accuracy:** $0.76$
* **Weighted F1:** $0.77$
