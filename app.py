import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("car_data.csv")

data = load_data()

st.title("Car Price Prediction App")

# Create features and target variable
X = data[['Year', 'Present_Price', 'Kms_Driven', 'Owner', 'Fuel_Type', 'Seller_Type', 'Transmission']]
y = data['Selling_Price']

# Perform one-hot encoding on categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model to disk
filename = 'LinearRegressionModel.pkl'
pickle.dump(model, open(filename, 'wb'))

# Load the model from disk
model = pickle.load(open(filename, 'rb'))

# Take input from user
year = st.slider("Select the Year of the Car", min_value=2003, max_value=2019, step=1)
present_price = st.slider("Select the Present Price (in Lakhs)", min_value=0.5, max_value=30.0, step=0.5)
kms_driven = st.slider("Select the Kilometers Driven", min_value=5000, max_value=50000, step=1000)
owner = st.slider("Select the Number of Previous Owners", min_value=0, max_value=2, step=1)

fuel_type = st.selectbox("Select Fuel Type", ['Petrol', 'Diesel'])
if fuel_type == 'Petrol':
    fuel_type_petrol = 1
    fuel_type_diesel = 0
else:
    fuel_type_petrol = 0
    fuel_type_diesel = 1

seller_type = st.selectbox("Select Seller Type", ['Individual', 'Dealer'])
if seller_type == 'Individual':
    seller_type_individual = 1
else:
    seller_type_individual = 0

transmission = st.selectbox("Select Transmission Type", ['Manual', 'Automatic'])
if transmission == 'Manual':
    transmission_manual = 1
else:
    transmission_manual = 0

# Make prediction
prediction = model.predict([[year, present_price, kms_driven, owner, fuel_type_petrol, fuel_type_diesel, seller_type_individual, transmission_manual]])

st.write(f"The estimated selling price of the car is {prediction[0]} Lakhs.")
