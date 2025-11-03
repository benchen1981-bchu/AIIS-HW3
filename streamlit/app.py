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
