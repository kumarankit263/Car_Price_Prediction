import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
model = pickle.load(open('LinearRegressionModel.pkl','rb'))

# Load the DataFrame
df = pickle.load(open('df.pkl','rb'))  # Load the DataFrame

st.title("Car Price Predictor")

# Select car name
name = st.selectbox('Car Name', df['name'].unique())

# Select car company
company = st.selectbox('Company', df['company'].unique())

# Select year
year = st.number_input('Year', min_value=1900, max_value=2024, step=1, value=2010)

# Enter kilometers driven
kms_driven = st.number_input('Kilometers Driven')

# Select fuel type
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])

if st.button('Predict Price'):
    # Preprocess input
    # You may need to encode categorical variables and scale numerical features here based on your preprocessing steps

    # Create query array
    query = np.array([[name, company, year, kms_driven, fuel_type]])
    query_df = pd.DataFrame(query, columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Assuming you have preprocessing steps to transform the query data similar to how it was done during training
    # You might need to encode categorical variables and scale numerical features accordingly

    # Make prediction
    prediction = model.predict(query_df)

    st.title("The predicted price of this car configuration is $" + str(prediction[0]))
