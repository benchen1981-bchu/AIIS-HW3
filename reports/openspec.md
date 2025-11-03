# OpenSpec: Spam Email Classification System

**Objective:**  
Develop an open, reproducible spam/ham email classifier with Streamlit UI for testing and hyperparameter tuning.

**Framework:** CRISP-DM

## 1. Business Understanding
- Detect spam emails automatically to improve security.

## 2. Data Understanding
- Data source: SMS Spam Collection dataset (UCI / Kaggle)
- Downloaded automatically by `src/download_data.py`

## 3. Data Preparation
- Clean text, remove stopwords, TF-IDF vectorization.

## 4. Modeling
- Models: Naive Bayes, Logistic Regression, SVM.
- Hyperparameters exposed in UI.

## 5. Evaluation
- Metrics: Accuracy, Precision, Recall, F1.

## 6. Deployment
- Streamlit web app.

## 7. Future Extensions
- Phishing detection, deep learning (LSTM), email header analysis.
