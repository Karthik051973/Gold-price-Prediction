import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Title
# -------------------------------
st.title("💰 Gold Price Prediction App")
st.markdown("Enter feature values to predict **Gold Price (GLD)**.")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("gld_price_data.csv")
    return data

gold_data = load_data()

# Features & Target
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Train-Test Split & Model Training
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, Y_train)

# -------------------------------
# User Input Prediction
# -------------------------------
st.subheader("🔮 Predict Gold Price")
st.markdown("Enter values for each feature:")

input_data = []
for col in X.columns:
    val = st.number_input(f"Enter {col}:", value=float(X[col].mean()))
    input_data.append(val)

if st.button("Predict Gold Price"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = regressor.predict(input_array)
    st.success(f"Predicted Gold Price: {prediction[0]:.2f}")
