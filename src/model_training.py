from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
import os

def build_pipeline(model_type='nb', **params):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    if model_type == 'nb':
        clf = MultinomialNB(**params)
    elif model_type == 'lr':
        clf = LogisticRegression(max_iter=1000, **params)
    elif model_type == 'svm':
        clf = SVC(probability=True, **params)
    else:
        raise ValueError("Unsupported model type")

    return Pipeline([
        ('tfidf', vectorizer),
        ('clf', clf)
    ])

def train_and_save(X_train, y_train, model_type='nb', save_path='../models'):
    os.makedirs(save_path, exist_ok=True)
    pipe = build_pipeline(model_type)
    pipe.fit(X_train, y_train)
    model_path = os.path.join(save_path, f"{model_type}_model.joblib")
    joblib.dump(pipe, model_path)
    print(f"Model saved to {model_path}")
    return pipe
