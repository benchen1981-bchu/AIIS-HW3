python src/download_data.py      # gets data from GitHub repo
python -m src.data_preparation   # preprocesses text
python -m src.model_training     # trains model(s)
streamlit run streamlit_app.py   # launches UI
