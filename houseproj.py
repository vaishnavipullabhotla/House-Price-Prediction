import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import warnings
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv(r'/Users/meghanagudise/Downloads/Housing.csv')

# Check for any null values
print(df.isna().sum())

# Check for duplicate rows
print(df.duplicated().any())

# Encoding categorical columns
lab_enc = LabelEncoder()
col = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
for i in col:
    df[i] = lab_enc.fit_transform(df[i])

print(df.head())

# Plot the heatmap to visualize correlations
'''
plt.figure(figsize=(10,12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
'''
# Features and target
X = df.drop('price', axis=1)
y = df['price']

# Standardization of features only (exclude target 'price')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and the scaler
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load the trained model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to get user input
def get_user_input():
    area = float(input("Enter the area of the house (in sq. ft.): "))
    bedrooms = int(input("Enter the number of bedrooms: "))
    bathrooms = int(input("Enter the number of bathrooms: "))
    stories = int(input("Enter the number of stories: "))
    mainroad = int(input("Is there a main road? (1 for Yes, 0 for No): "))
    guestroom = int(input("Is there a guest room? (1 for Yes, 0 for No): "))
    basement = int(input("Is there a basement? (1 for Yes, 0 for No): "))
    hotwaterheating = int(input("Is there hot water heating? (1 for Yes, 0 for No): "))
    airconditioning = int(input("Is there air conditioning? (1 for Yes, 0 for No): "))
    parking = int(input("Enter the number of parking spaces: "))
    prefarea = int(input("Is there a preferred area? (1 for Yes, 0 for No): "))
    furnishingstatus = int(input("Enter furnishing status (0: Unfurnished, 1: Semi-furnished, 2: Furnished): "))
    
    return {
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

# Get new data from the user
new_data = get_user_input()

# Convert the new input data to a DataFrame
new_input_df = pd.DataFrame([new_data])

# Normalize the new input data using the same scaler (no 'price' column)
new_input_scaled = scaler.transform(new_input_df)

# Predict the price
predicted_price = model.predict(new_input_scaled)

print("The predicted price of the house is:", predicted_price[0])
