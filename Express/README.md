
# SBA Loan Eligibility Prediction

This repository contains notebooks and models to generate synthetic SBA loan data and predict loan eligibility using machine learning. The project simulates business loan scenarios and classifies them as eligible for SBA 7(a), SBA 504 or SBA Express programs.

---

## 📁 Files

### `Data_Generation.ipynb`
- **Purpose**: Generates a synthetic dataset simulating realistic business loan applications.
- **Highlights**:
  - Uses randomized logic to create fields like:
    - Annual revenue over 3 years
    - Debt-to-income ratios
    - Business acquisition flags
    - DSCR (Debt Service Coverage Ratio)
    - Loan amount and eligibility tags (`7(a)`, `504` and `SBA Express`)
  - Exports data for model training

### `Express.ipynb`
- **Purpose**: Loads the generated dataset, preprocesses it, and trains machine learning models to predict loan eligibility.
- **Pipeline**:
  - Cleans and encodes features
  - Trains multiple classifiers (Logistic Regression, Random Forest, VotingClassifier)
  - Evaluates with precision-recall curves and confusion matrices
  - Saves the best-performing model (`voting_model.pkl`)

### `voting_model.pkl`
- **Purpose**: Pre-trained Voting Classifier model for real-time SBA eligibility predictions.
- **Includes**:
  - Logistic Regression
  - Random Forest
  - K-Nearest Neighbors (or other classifiers used)

---

[Parlay Docs Drive Link |https://drive.google.com/drive/folders/16KTGAEswErWP6BlLL1UjNlbbRUZogexa]

## 🧠 Key Insights

- A precision-recall tradeoff analysis reveals:
  - **Low thresholds** give high recall (catch more Express loans) but poor precision (more false positives).
  - Optimal threshold tuning is essential depending on business objectives.

- **Note on Testing Data**:
  - In realistic scenarios, the dataset will be **heavily imbalanced**.
  - SBA Express loan eligibility (label = 1) occurs much less frequently than ineligibility (label = 0).
  - Therefore, performance metrics like precision, recall, and F1-score might show **absurd values** if the model is tested on real or skewed datasets.

---


## 🧰 Requirements

- Python 3.8+
- pandas
- scikit-learn
- seaborn
- matplotlib
- joblib

---

## 📌 Author

Sai Preetham Vinnakota  
_M.S. in Data Science, University of Colorado Boulder_
