# House Price Prediction

Predicts house prices using a Linear Regression model trained on housing data.  
Built with **Python, Scikit-learn, Pandas, and Streamlit**.

## Features
- Train model on dataset (`houseproj.py`)
- Web-based prediction app (`house_price_prediction.py`)
- Preprocessing with StandardScaler & Label Encoding
- Saved model & scaler for reuse

## Files
- `houseproj.py` – Train & save model
- `house_price_prediction.py` – Streamlit UI for predictions
- `house_price_model.pkl` – Trained model
- `scaler.pkl` – Preprocessing scaler
- `Housing.csv` – Dataset

## Run App
```bash
pip install -r requirements.txt
streamlit run house_price_prediction.py
