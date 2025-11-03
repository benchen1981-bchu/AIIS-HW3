import streamlit as st
import joblib
from spam_classifier import build_pipeline
from src.data_preparation import load_raw_data, preprocess_and_split

st.title("Spam Email Classifier")

model_name = st.selectbox("Select model", ["Naive Bayes", "Logistic Regression", "SVM"])
# Hyper-parameter inputs:
if model_name == "Naive Bayes":
    alpha = st.slider("Alpha (smoothing)", 0.1, 2.0, 1.0, 0.1)
    params = {'clf__alpha': alpha}
elif model_name == "Logistic Regression":
    C = st.slider("C (inverse regularization)", 0.01, 10.0, 1.0, 0.01)
    params = {'clf__C': C, 'clf__solver': 'liblinear'}
elif model_name == "SVM":
    C = st.slider("C", 0.01, 10.0, 1.0, 0.01)
    kernel = st.selectbox("Kernel", ["linear", "rbf"])
    params = {'clf__C': C, 'clf__kernel': kernel, 'clf__probability': True}

message = st.text_area("Enter email message (subject + body)")

if st.button("Classify"):
    # Build pipeline with selected model + params:
    if model_name == "Naive Bayes":
        pipe = build_pipeline(model_name='nb', **params)
    elif model_name == "Logistic Regression":
        pipe = build_pipeline(model_name='lr', **params)
    else:
        pipe = build_pipeline(model_name='svm', **params)

    # Load data and train the model
    df = load_raw_data()
    X_train, _, y_train, _ = preprocess_and_split(df)
    pipe.fit(X_train, y_train)

    pred = pipe.predict([message])[0]
    prob = pipe.predict_proba([message])[0]
    st.write("Prediction:", pred)
    st.write("Probability:", prob)