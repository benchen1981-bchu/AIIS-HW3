import pandas as pd
import os
import re
import nltk
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
from nltk.corpus import stopwords

def load_raw_data():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "SMSSpamCollection")
    df = pd.read_csv(path, sep='\t', names=['label', 'text'])
    return df

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    stop = set(stopwords.words('english'))
    text = " ".join([w for w in text.split() if w not in stop])
    return text

def preprocess_and_split(df):
    df['clean_text'] = df['text'].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
    return X_train, X_test, y_train, y_test
