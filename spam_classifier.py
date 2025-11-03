import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_data(path):
    df = pd.read_csv(path)
    # assume columns: “text”, “label”
    return df

def preprocess_and_split(df, test_size=0.2, random_state=42):
    X = df['text']
    y = df['label']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def build_pipeline(model_name='nb', **kwargs):
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    if model_name == 'nb':
        model = MultinomialNB(**kwargs)
    elif model_name == 'lr':
        model = LogisticRegression(max_iter=1000, **kwargs)
    elif model_name == 'svm':
        model = SVC(probability=True, **kwargs)
    else:
        raise ValueError("Unsupported model name")
    pipe = Pipeline([
        ('tfidf', vec),
        ('clf', model)
    ])
    return pipe

def tune_model(pipe, param_grid, X_train, y_train):
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_macro', verbose=2, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

def save_model(model, path='model.joblib'):
    joblib.dump(model, path)

def load_model(path='model.joblib'):
    return joblib.load(path)

if __name__ == "__main__":
    df = load_data('spam_dataset.csv')
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    pipe = build_pipeline(model_name='nb')
    param_grid = {
        'clf__alpha': [0.5, 1.0, 1.5]
    }
    best_model, best_params = tune_model(pipe, param_grid, X_train, y_train)
    print("Best params:", best_params)
    evaluate_model(best_model, X_test, y_test)
    save_model(best_model)
