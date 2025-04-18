# Parlay Team 1

## Table of Contents

- [About](#-about)
- [Progress](#-progress)
- [MissingFiles](#-missingfiles)
- [Data](#-data)
- [Code](#-code)
- [Output](#-output)


  
## About

This is a collaborative capstone project between Parlay Finance and the Data Science major at the University of Colorado Boulder. All data belongs to Parlay and may not be used without authorization.
For the Spring 2025 semester, Team 1 and Parlay focused on what could improve the accuracy of the prediction of the loan possibility.





## 🚀 Progress

### What we've done 
- [x] Generate Synthetic Datasets
- [x] Machine Learning Several Models per Loan Types
- [x] Test models with the randomly generated datasets.

### What Should be the Next Step
- [ ] Test Models with the real Dataset
- [ ] Check the accuracy of the models
- [ ] If Models are inaccurate, WHY?
- [ ] If Models works well, What should we do?

## 📝 Data

### Data_Generataion.ipynb
This is the main synthetic data generation file. 
Including all the rules about data relationships and custom part. 

to generate the data, find 

```shell
df = generate_corrected_synthetic_data(100000)
```

and change the number to desired size.


### Synthetic_SBA_Loans_raw.csv
This is raw generation file with 100k rows.

### Synthetic_SBA_Loans.csv
This is after sorting out only 7(a) eligible rows by following provided rule from Parlay

### Synthetic_SBA_Loans_test.csv
This is small raw generation file for testing.

### Predicted_SBA_Loans_raw.csv
After processing via Predict.ipynb, now test file has prediction from each models.



## 📝 Code


### NN_and_Meta / Model_Generation.ipynb
This is the code making the models. For this version, model is trained with 32 features. For further research, 

```shell
selected_columns = [
    'Business Ownership (1)', 'Business Ownership (2)', 'Business Ownership (3)',
    'Business Ownership (4)', 'Business Ownership (5)', 'Annual Revenue (2 years ago)',
    'Annual Revenue (1 year ago)', 'Annual Revenue (latest year)', 'Net Profit Margin',
    'Business Debt (2 years ago)', 'Business Debt (1 year ago)', 'Business Debt (latest year)',
    'NOI (2 years ago)', 'NOI (1 year ago)', 'NOI (latest year)', 'Managerial Experience',
    'Years in Business', 'Collateral Availability', 'Acquisition Request', 'Working Capital',
    'Business Expansion', 'Equipment Purchase or Leasing', 'Inventory Purchase',
    'Real Estate Acquisition or Improvement', 'Business Acquisition or Buyout',
    'Refinancing Existing Debt', 'Emergency Funds', 'Franchise Financing',
    'Contract Financing', 'Licensing or Permits', 'Line of Credit Establishment',
    'Eligibility Score'
]
```

Just need to add or change the desired column you want.

### Predict.ipynb
This will add model prediction result to the each rows. For currently, CatBoost, GradientBoost, LightGBM, Logistic Regression, Random Forest, XGBoost, Neural Nets and Meta Learning was used for the models.


## 📚 Missing Files
Because generated datas are way to large, this needs to be download manualy. 

[Google Drive Link]([https://drive.google.com/drive/folders/1vMSg__xCttENbtfuT4msyWSCoJucCyKO?usp=sharing](https://drive.google.com/drive/folders/1vMSg__xCttENbtfuT4msyWSCoJucCyKO?usp=sharing))

There are 2 csv files, 
- Synthetic_SBA_Loans_raw.csv
  - This is the raw file which has 100k raw data
  - Location is, "Junsoo_Jung_Work/Synthetic_SBA_Loans_raw.csv"
- Synthetic_SBA_Loans.csv
  - This is cleaned version, which has only 7a pre-approved.
  - Location is, "Junsoo_Jung_Work/Synthetic_SBA_Loans.csv"

And one Pickle model.
- RandomForest_model.pkl
  - Random Forest model.
  - Location is, "Junsoo_Jung_Work/saved_model/RandomForest_model.pkl"



## Output

For the final output, see "Junsoo_Jung_Work/Predicted_SBA_Loans.csv"
Columns named (Model)_Prediction shows results of each model's prediction.

