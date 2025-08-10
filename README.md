# 💳 Credit Card Fraud Detection System

This project is a machine learning-based system designed to intelligently detect fraudulent credit card transactions. It uses real-world data and multiple classification algorithms to identify patterns commonly associated with fraud.

## 🔍 Project Objectives

- Preprocess and clean a real credit card transactions dataset
- Train and compare multiple machine learning models
- Evaluate each model using metrics like ROC AUC, Precision, Recall, and F1-score
- Select and save the best-performing model
- Build an interactive web interface using Streamlit for real-time fraud prediction

## 📊 Dataset

- **Source**: Publicly available dataset from Kaggle  
  [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains anonymized transaction features (`V1` to `V28`), `Amount`, and `Class` (fraud or not)

## 🧠 Machine Learning Models Used

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Neural Network (MLPClassifier)

## 🧪 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- ROC AUC Score

The best model is selected based on the highest ROC AUC score and saved for deployment.

## 🖥️ Web App Interface

Built using **Streamlit**, the app allows users to input transaction details (V1–V28, Amount) and get a prediction:
- **0** → Not Fraud
- **1** → Fraud  
With confidence level shown.
