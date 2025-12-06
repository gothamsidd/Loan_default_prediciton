# 🧠 Loan Default Prediction using Machine Learning
A complete end-to-end machine learning pipeline for predicting loan default using the **Kaggle Home Credit Default Risk** dataset.  
This project focuses on **data preprocessing, handling missing values, feature engineering, exploratory analysis, model benchmarking, and evaluation under severe class imbalance**.

---

## 📌 Project Overview
Financial institutions rely on credit scoring models to estimate the likelihood that a borrower will default on a loan.  
The **Home Credit Default Risk** dataset provides extensive demographic, behavioral, and financial information about loan applicants.

This project implements a baseline machine learning workflow to:
- Clean and preprocess raw data  
- Engineer meaningful predictive features  
- Analyze feature relationships  
- Train benchmark models  
- Evaluate model performance under class imbalance  

---

## 📂 Repository Structure

Loan_Default_Prediction/
│
├── loan_default.py # Main ML pipeline script
├── README.md # Project documentation
│
├── figures/ # Visualization plots
│ ├── ext_sorces.png
│ ├── corr_heatmap.png
│ ├── confusion_log.png
│ └── roc_comparison.png
│
└── notebooks/ # (Optional) Jupyter notebooks for exploration


## Dataset
**Source:** Kaggle — Home Credit Default Risk  
**Size:** 307,511 rows × 122 features  
**Challenges:**  
- High missing values  
- Anomalous values (DAYS_EMPLOYED = 365243)  
- Mixed data types  
- Severe class imbalance (~8% defaults)  
Dataset link: https://www.kaggle.com/c/home-credit-default-risk

## Methodology

### 1. Preprocessing
- Removed columns with >40% missingness  
- Replaced anomalous DAYS_EMPLOYED values with NaN  
- Median imputation for numeric features  
- Label encoding + “MISSING” category for categoricals  
- IQR outlier capping for financial variables  
- Standardization using StandardScaler  

### 2. Feature Engineering
Created the following derived features:  
- AGE  
- EMPLOY_YEARS  
- CREDIT_INCOME_RATIO  
- ANNUITY_INCOME_RATIO  
- EXT_SOURCES_MEAN  

### 3. Modeling
Models evaluated:  
- Logistic Regression  
- K-Nearest Neighbors (subset)  
- Decision Tree (depth 6)  
Train/Test Split: **80/20 stratified**  
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC  

## Results

### Performance Summary
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.9194 | 0.5500 | 0.0066 | 0.0131 | **0.743** |
| KNN (subset)        | 0.9126 | 0.2619 | 0.0264 | 0.0479 | 0.5679 |
| Decision Tree       | 0.9193 | 0.4545 | 0.0010 | 0.0020 | 0.7179 |

### Insights
- Models show **very low recall** due to dataset imbalance.  
- Logistic Regression provides the best ranking ability (AUC = 0.743).  
- EXT_SOURCE_2 and EXT_SOURCE_3 are the strongest predictors.  

## Visualizations
Stored in `figures/`:
- **ext_sorces.png** — External credit score distribution  
- **corr_heatmap.png** — Feature correlation heatmap  
- **confusion_log.png** — Logistic Regression confusion matrix  
- **roc_comparison.png** — ROC curve comparison  

## Limitations
- Models not tuned or optimized  
- No SMOTE or class rebalancing  
- KNN inefficient for large datasets  
- Default threshold (0.5) unsuitable for imbalance  

## Future Work
- Apply SMOTE, ADASYN, class weighting  
- Hyperparameter optimization  
- Use advanced models (XGBoost, LightGBM, Random Forests)  
- Add SHAP explainability  
- Build deployment-ready API or web app  

## Running the Project
1.Clone the repo
```git clone https://github.com/gothamsidd/Loan_default_prediciton```
```cd Loan_default_prediciton```
2.Install dependencies:
```pip install -r requirements.txt```
Run the script:
```python loan_default.py```


## Contributors

This project was developed as part of an academic machine-learning study.

Siddharth Pareek — siddharth.pareek@adypu.edu.in

Rounit Singh — rounit.singh@adypu.edu.in



## License
This project is intended for academic and research purposes only.
You may view, modify, and extend the code for learning or experimentation.
Commercial redistribution or deployment is not permitted without permission.



  


























