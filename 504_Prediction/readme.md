# Loan Eligibility Meta-Model (504 SBA) - Machine Learning Pipeline

This repository contains a machine learning project for predicting **loan eligibility** under the **504 Small Business Administration (SBA) program** using an ensemble model (meta-learner). The notebook `ml_model_504.ipynb` outlines the complete workflow from data preprocessing to model stacking and evaluation.

---

## 📁 Dataset

- `Synthetic_Loans_Data1.csv`: Training data containing loan applicant details and eligibility labels.
- `Synthetic_Loans_Data_val.csv`: Validation dataset for model performance evaluation.

---

## Project Workflow Overview

### 1. Data Loading & Cleaning
- Load training and validation datasets.
- Drop irrelevant or identifier columns like `Applicant ID` and `Location`.
- Convert boolean columns to integers to prepare for ML modeling.

### 2. Class Balancing
- Perform **undersampling** to address class imbalance between eligible (minority) and ineligible (majority) applicants.

### 3. Feature Engineering
- Apply **`pd.get_dummies()`** to handle categorical variables.
- Use **`align()`** to ensure the same feature space between training and validation data.

### 4. Data Normalization for Neural Network
- Identify numeric columns and scale them using `StandardScaler`.
- Save scaled versions for use with the neural network.

---

## Models Trained

Three base models are trained independently:

| Model          | Purpose                        | Framework |
|----------------|--------------------------------|-----------|
| Neural Network | Learns complex non-linear patterns | TensorFlow / Keras |
| XGBoost        | High-performing gradient boosting tree | XGBoost |
| LightGBM       | Fast and efficient boosting model | LightGBM |

---

## Meta-Learner

A **Logistic Regression** model is used as a **meta-learner**. It learns to combine predictions from the base models to make a final eligibility decision.

---

## Evaluation

The meta-learner is evaluated on the validation set using:
- Accuracy
- ROC AUC Score
- Confusion Matrix

---

## Streamlit Deployment (Coming Soon)

You can deploy this model stack using `Streamlit` for interactive predictions. The app will:
- Accept user inputs for loan features.
- Preprocess them (including dummy encoding and scaling).
- Feed them into the 3 base models.
- Combine their outputs via the meta-learner.
- Display the final eligibility result.

---

## Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm tensorflow joblib streamlit
```
## Author
  Venkata Harshith Nikhil Samudrala
  
  M.S. Data Science, University of Colorado Boulder
