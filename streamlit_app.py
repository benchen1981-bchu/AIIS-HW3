import streamlit as st
#import joblib
from joblib import Parallel, delayed
from joblib import Memory

from spam_classifier import load_model, build_pipeline

st.title("Spam Email Classifier")

model_name = st.selectbox("Select model", ["Naive Bayes", "Logistic Regression", "SVM"])
# Hyper‐parameter inputs:
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
    # Load pre-trained model? Or train quickly? Better: load a base model, then fine-tune?
    # For simplicity: load a saved model for each type; or train on the fly (but may be slow).
    model = pipe.fit(...)  # or load
    pred = pipe.predict([message])[0]
    prob = pipe.predict_proba([message])[0]
    st.write("Prediction:", pred)
    st.write("Probability:", prob)

# Optionally: Show evaluation metrics of base model


##
streamlit_app.py

import streamlit as st
import joblib
from src.model_training import build_pipeline
from src.data_preparation import clean_text
import os

st.title("📧 Spam Email Classifier")
st.markdown("Adjust hyperparameters, input a message, and test classification results.")

model_type = st.selectbox("Select model", ["Naive Bayes", "Logistic Regression", "SVM"])

# Hyperparameters
params = {}
if model_type == "Naive Bayes":
    params['alpha'] = st.slider("Alpha", 0.1, 2.0, 1.0, 0.1)
elif model_type == "Logistic Regression":
    params['C'] = st.slider("C", 0.01, 5.0, 1.0)
elif model_type == "SVM":
    params['C'] = st.slider("C", 0.01, 5.0, 1.0)
    params['kernel'] = st.selectbox("Kernel", ["linear", "rbf"])

text_input = st.text_area("Enter your email message here:")

if st.button("Classify"):
    msg = clean_text(text_input)
    pipe = build_pipeline(model_type.lower().split()[0], **params)
    model_path = os.path.join("models", f"{model_type.lower().split()[0]}_model.joblib")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.warning("Model not found. Please train first using the training script.")
        st.stop()

    pred = model.predict([msg])[0]
    prob = model.predict_proba([msg])[0].max()

    st.success(f"**Prediction:** {pred.upper()} (Confidence: {prob:.2f})")

