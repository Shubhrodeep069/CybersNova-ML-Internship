# 💳 Credit Card Fraud Detection

## 📌 Project Overview

This project develops a machine learning system to detect fraudulent credit card transactions.

The project focuses on handling severe class imbalance and evaluating the model using fraud-detection metrics such as Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

## 🎯 Objectives

- Analyze credit card transaction data
- Identify class imbalance
- Preprocess and scale transaction features
- Compare different classification approaches
- Apply SMOTE for imbalance handling
- Train a Random Forest fraud detection model
- Optimize the classification threshold
- Build a Streamlit prediction interface

## 📊 Dataset

Dataset: Credit Card Fraud Detection

Source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains 284,807 transactions, including 492 fraudulent transactions.

The features include:
- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

`Class = 0` represents a normal transaction.

`Class = 1` represents a fraudulent transaction.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## 🤖 Models Tested

1. Logistic Regression
2. Class-Weighted Logistic Regression
3. SMOTE + Logistic Regression
4. Random Forest

## 📈 Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 99.92% | 82.89% | 64.29% | 72.41% | 95.59% |
| Class-Weighted Logistic Regression | 97.55% | 6.10% | 91.84% | 11.44% | 97.22% |
| SMOTE + Logistic Regression | 97.42% | 5.80% | 91.84% | 10.92% | 96.99% |
| Random Forest | 99.95% | 90.59% | 78.57% | 84.15% | 95.73% |

## 🏆 Final Model

Random Forest was selected as the final model because it provided the best balance between precision and recall.

A classification threshold of **0.40** was selected after threshold analysis.

### Final Threshold Performance

- Precision: 88.04%
- Recall: 82.65%
- F1-Score: 85.26%

### Final Confusion Matrix

| | Predicted Normal | Predicted Fraud |
|---|---:|---:|
| Actual Normal | 56,853 | 11 |
| Actual Fraud | 17 | 81 |

## 🔍 Feature Importance

The most important features identified by the Random Forest model were:

1. V14
2. V10
3. V4
4. V12
5. V17
6. V3
7. V11
8. V16
9. V2
10. V9

The V1–V28 features are PCA-transformed features, so they do not have direct human-readable meanings.

## 🖥️ Streamlit Application

The project includes a Streamlit web application that allows users to:

- Enter transaction information
- Load sample fraud transactions
- Load sample normal transactions
- Calculate fraud probability
- Classify transactions as Normal or Fraud

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── random_forest_fraud_model.pkl
│   └── transaction_scaler.pkl
│
├── notebooks/
│   └── Credit_Card_Fraud_Detection.ipynb
│
├── app.py
├── requirements.txt
└── README.md


▶️ How to Run
1. Clone the repository
git clone <your-repository-url>
cd Credit-Card-Fraud-Detection
2. Create virtual environment
python -m venv venv
3. Activate virtual environment

Windows:

venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
5. Run the Streamlit application
streamlit run app.py
⚠️ Note

The dataset contains anonymized PCA-transformed features. The model is intended for educational and demonstration purposes and should not be considered a production financial fraud detection system without additional validation, monitoring, security, and real-world testing.

👨‍💻 Author

Shubhrodeep Majumder
