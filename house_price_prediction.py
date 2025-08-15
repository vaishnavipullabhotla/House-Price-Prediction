import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict house price
def predict_price(new_data):
    # Convert input data to a DataFrame
    new_input_df = pd.DataFrame([new_data])

    # Normalize the input data using the same scaler
    new_input_scaled = scaler.transform(new_input_df)

    # Predict the price
    predicted_price = model.predict(new_input_scaled)
    return predicted_price[0]

# Streamlit App
st.set_page_config(page_title="House Price Prediction", layout="centered", initial_sidebar_state="collapsed")

# Title of the app
st.title("House Price Prediction App")

# Neutral background color and clean format
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Input fields for the features
st.subheader("Enter the house features below:")

area = st.number_input("Area of the house (in sq. ft.):", min_value=500.0, max_value=15000.0, step=100.0, value=1000.0)
bedrooms = st.number_input("Number of bedrooms:", min_value=1, max_value=10, step=1, value=3)
bathrooms = st.number_input("Number of bathrooms:", min_value=1, max_value=10, step=1, value=2)
stories = st.number_input("Number of stories:", min_value=1, max_value=5, step=1, value=2)
mainroad = st.selectbox("Is there a main road?", options=["Yes", "No"], index=0)
guestroom = st.selectbox("Is there a guest room?", options=["Yes", "No"], index=0)
basement = st.selectbox("Is there a basement?", options=["Yes", "No"], index=1)
hotwaterheating = st.selectbox("Is there hot water heating?", options=["Yes", "No"], index=1)
airconditioning = st.selectbox("Is there air conditioning?", options=["Yes", "No"], index=0)
parking = st.number_input("Number of parking spaces:", min_value=0, max_value=5, step=1, value=1)
prefarea = st.selectbox("Is there a preferred area?", options=["Yes", "No"], index=0)
furnishingstatus = st.selectbox("Furnishing status", options=["Unfurnished", "Semi-furnished", "Furnished"], index=1)

# Convert categorical inputs to 0/1 values
mainroad = 1 if mainroad == "Yes" else 0
guestroom = 1 if guestroom == "Yes" else 0
basement = 1 if basement == "Yes" else 0
hotwaterheating = 1 if hotwaterheating == "Yes" else 0
airconditioning = 1 if airconditioning == "Yes" else 0
prefarea = 1 if prefarea == "Yes" else 0
furnishingstatus_mapping = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}
furnishingstatus = furnishingstatus_mapping[furnishingstatus]

# Create a dictionary for the user input data
new_data = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'mainroad': mainroad,
    'guestroom': guestroom,
    'basement': basement,
    'hotwaterheating': hotwaterheating,
    'airconditioning': airconditioning,
    'parking': parking,
    'prefarea': prefarea,
    'furnishingstatus': furnishingstatus
}

# Predict button
if st.button("Predict Price"):
    # Call the prediction function
    predicted_price = predict_price(new_data)
    
    # Display the predicted price in a neat format
    st.success(f"The predicted price of the house is: ₹{predicted_price:,.2f}")