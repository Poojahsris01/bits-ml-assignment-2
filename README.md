# BITS Pilani - WILP Division
## M.Tech (AIML) - Machine Learning (Assignment - 2)

**Student Name:** Poojahsri Sivakumar  
**BITS ID:** 2026ak05655  

---

### Mandatory Links
- **GitHub Repository Link:** https://github.com/Poojahsris01/bits-ml-assignment-2
- **Live Streamlit App Link:** https://bits-ml-assignment-2-ngysds348d3356m9cahsuo.streamlit.app/

---

### A. Problem Statement
Prediction of whether a credit card client will default on their payment in the following month, based on 23 demographic, credit-limit, repayment-history, bill-amount and payment-amount attributes, using 5 different machine learning classification algorithms. The target is imbalanced (about 22% defaulters), so the models are compared on AUC, Precision, Recall, F1 and MCC as well as Accuracy.

---

### B. Dataset Description
- **Dataset Source:** UCI Machine Learning Repository - Default of Credit Card Clients (Taiwan, 2005)
- **Total Instances:** 30,000 rows originally; Rows remaining after removing duplicates 29,965
- **Total Features:** 23 input features
  - Demographic: `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
  - Credit limit: `LIMIT_BAL`
  - Repayment status (Apr-Sep): `PAY_1` to `PAY_6`
  - Bill statement amounts: `BILL_AMT1` to `BILL_AMT6`
  - Previous payment amounts: `PAY_AMT1` to `PAY_AMT6`
- **Target Variable:** `default_next_month` (Binary Classification: 0 = No default, 1 = Default)
- **Class Distribution:** about 78% No default / 22% Default (imbalanced)
- **Preprocessing:**
  - Undocumented `EDUCATION` codes (0, 5, 6) merged into the "others" category (4)
  - Undocumented `MARRIAGE` code 0 merged into the "others" category (3)
  - Duplicate rows removed so copies cannot appear in both train and test sets
  - `PAY_` columns kept as ordinal repayment-status codes
  - `StandardScaler` applied (inside a Pipeline saved with the model) for Logistic Regression, KNN and SVM
- **Train/Test Split:** 75% train / 25% test (22,473 train rows, 7,492 test rows), stratified on the target, random seed 8531

---

### C. Github Repository Link
- **GitHub Repository Link:** https://github.com/Poojahsris01/bits-ml-assignment-2
- **Live Streamlit App Link:** https://bits-ml-assignment-2-ngysds348d3356m9cahsuo.streamlit.app/


**Repository contents:**
```
├── credit_default_models.ipynb   (data cleaning, training, evaluation)
├── app.py                        (Streamlit application)
├── requirements.txt
├── README.md
├── test_data.csv                 (held-out test data)
├── model_comparison.csv
└── model/                        (saved .joblib models for all five classifiers)
```

---

### D. Models Used and Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.6945 | 0.7352 | 0.3889 | 0.6665 | 0.4912 | 0.3156 |
| **Decision Tree** | 0.7419 | 0.7668 | 0.4445 | 0.6665 | 0.5333 | 0.3789 |
| **KNN** | 0.8125 | 0.7468 | 0.6511 | 0.3287 | 0.4369 | 0.3672 |
| **Naive Bayes** | 0.3684 | 0.6704 | 0.2447 | 0.8884 | 0.3837 | 0.1139 |
| **Support Vector Machine** | 0.7835 | 0.7711 | 0.5092 | 0.6001 | 0.5509 | 0.4120 |

Precision, Recall and F1 are computed for the positive class (Default).

---

### Model Observations on the performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Balanced class weights give a good recall (0.6665) but low precision (0.3889), so many non-defaulters are flagged. The linear boundary limits its AUC (0.7352) and MCC (0.3156). It is fast and interpretable but cannot capture non-linear patterns in the repayment history. |
| **Decision Tree** | With depth limited to 6, it beats Logistic Regression on accuracy, precision, AUC, F1 and MCC (AUC 0.7668, F1 0.5333, MCC 0.3789) and ties it on recall. The repayment-status features likely drive its splits. It is easy to explain, but a single tree is less stable than an ensemble.|
| **KNN** | Highest accuracy (0.8125) and precision (0.6511) but the lowest recall (0.3287). Defaulters are the minority class, so neighbours mostly vote "no default", and KNN has no class-weight option. Accuracy flatters this model; it misses about two thirds of real defaulters. |
| **Naive Bayes** | Highest recall (0.8884) but very poor accuracy (0.3684), precision (0.2447) and MCC (0.1139). The independence assumption is violated because the six `BILL_AMT` columns are strongly correlated, so the model predicts "default" far too often. |
| **Support Vector Machine** | The RBF kernel captures non-linear relationships and gives the best AUC (0.7711), F1 (0.5509) and MCC (0.4120), with balanced recall (0.6001) and precision (0.5092). The trade-off is slower training and prediction and lower interpretability. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** - highest AUC (0.7711), F1 (0.5509) and MCC (0.4120), the metrics that remain reliable on an imbalanced target. The Decision Tree is a close and much faster runner-up. |

---

### Streamlit App Features
- CSV upload of test data (must contain all 23 features and `default_next_month`)
- Model selection dropdown (5 models)
- Display of Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix and classification report
- Side-by-side comparison of all models on the uploaded data
- Saved models are loaded directly; nothing is retrained on the uploaded data
