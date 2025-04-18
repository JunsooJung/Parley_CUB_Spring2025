# Streamlit App: 504 Loan Eligibility Predictor

This is a web-based Streamlit application that predicts **loan eligibility under the 504 SBA program** using a **stacked ensemble model** composed of:

- A Neural Network (Keras)
- XGBoost Model
- LightGBM Model
- Logistic Regression Meta-Learner

The model stack uses both numeric and categorical business financial information to predict whether a business is eligible (`1`) or not eligible (`0`) for a 504 loan.

---

## Features

- Interactive form-based input  
- Real-time prediction using 3 base models  
- Ensemble (meta-model) logic for more accurate decisions  
- Preprocessing includes one-hot encoding and numeric scaling  
- User-friendly web interface using Streamlit

---

## Model Architecture

| Model          | Role                  | Framework        |
|----------------|-----------------------|------------------|
| Neural Network | Feature pattern learning | TensorFlow / Keras |
| XGBoost        | Tree-based learner    | XGBoost          |
| LightGBM       | Efficient boosting    | LightGBM         |
| Meta-Model     | Combiner of predictions | Logistic Regression (sklearn) |

---


The app handles necessary preprocessing steps including **dummy encoding** and **scaling** of numeric inputs.

---

## 📦 Requirements

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm tensorflow joblib
