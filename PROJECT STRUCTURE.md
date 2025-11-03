# PROJECT STRUCTURE

spam_email_classifier/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── models/
│   └── (trained models saved here)
│
├── reports/
│   ├── openspec.md
│   └── model_metrics.json
│
├── src/
│   ├── __init__.py
│   ├── download_data.py
│   ├── data_preparation.py
│   ├── model_training.py
│   ├── evaluate_model.py
│   └── utils.py
│
├── streamlit_app.py
│
├── requirements.txt
│
└── README.md

-----------------------------
data/raw/Chapter04/
│
├── spam.csv
├── hamdata.txt
├── spamdata.txt
├── phishing_email.csv
└── related notebooks and assets

--------------------------------
spam_email_classifier/
├── data/raw/
├── src/
│ ├── download_data.py ← pulls data from Packt GitHub
│ ├── data_preparation.py
│ ├── model_training.py
│ ├── evaluate_model.py
├── streamlit_app.py
├── requirements.txt
├── reports/openspec.md
└── README.md
--------