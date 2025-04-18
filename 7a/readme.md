## Overview

This folder contains synthetic data, code, and model for analyzing and predicting eligibility for U.S. Small Business Administration (SBA) loan program specifically the 7(a) loan. This includes large-scale, realistic dataset, Jupyter notebooks for data generation and modeling, and a trained XGBoost model for eligibility prediction.

---

## Data Description

### Datasets

- **given.csv**:  
  Small, illustrative dataset with feature definitions and value ranges for each field. Used as a schema template for synthetic data generation.

- **Synthetic_SBA_Loans.csv**:  
  Large-scale, realistic synthetic dataset (thousands to millions of rows) with the following key columns:
    - Business demographics (structure, location, NAICS code, ownership)
    - Financial history (revenues, debts, NOI, DSCR, profit margins) for three years
    - Credit scores (personal and business)
    - Loan characteristics (amount, collateral, purposes)
    - Experience metrics (industry, managerial, years in business)
    - Binary flags for loan purposes (e.g., working capital, equipment, expansion)
    - Eligibility label for SBA loan programs

### Model

- **best_xgb.pkl**:  
  Pickled XGBoost classifier trained to predict SBA loan eligibility based on the synthetic dataset.  
  - Input: Features from the synthetic datasets  
  - Output: Eligibility prediction (binary or probability)

---

## Code & Notebooks

- **Data_Generation.ipynb**:  
  Generates synthetic loan applicant data using weighted random sampling, business logic, and regulatory rules (e.g., SBA 20% ownership rule, NAICS code filtering, realistic financials).

- **Clustering_Test.ipynb**:  
  Loads `Synthetic_SBA_Loans_more.csv` and applies clustering algorithms to segment applicants based on business and financial features.

- **XG_Boost.ipynb**:  
  Trains and evaluates an XGBoost model for predicting loan eligibility. Includes feature engineering, model selection, and performance metrics.

- **Score_Calculation_XG_Boost.ipynb**:  
  Applies the trained XGBoost model (`best_xgb.pkl`) to new data, calculates probabilities, and interprets results for downstream analysis.

---

## Data Generation & Assumptions

- **Data-generation-assumptions-and-initial-results.pdf**:  
  Comprehensive documentation of the data generation process, including:
    - Business structure and ownership logic
    - Credit and financial metric distributions
    - Loan amount and collateral rules
    - Geographic and industry eligibility
    - Experience and operational metrics
    - Loan purpose flag logic
    - Data validation and exception handling
    - Initial modeling results and key performance metrics[3]

---

## Usage

1. **Explore the data**
- Open and run `Data_Generation.ipynb` to generate or inspect synthetic data.
- Use `Clustering_Test.ipynb` and `XG_Boost.ipynb` for analysis and modeling.

2. **Model inference**
- Load `best_xgb.pkl` in your Python environment to predict eligibility on new data:
  ```
  import pickle
  with open('best_xgb.pkl', 'rb') as f:
      model = pickle.load(f)
  # model.predict(X) where X is a DataFrame with the required features
  ```
