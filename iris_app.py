# iris_app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    df.drop("Id", axis=1, inplace=True)
    return df

df = load_data()

# Train the model
X = df.drop("Species", axis=1)
y = df["Species"]
model = RandomForestClassifier()
model.fit(X, y)

# App Title
st.title("🌸 Iris Flower Predictor")
st.markdown("Enter the flower's measurements below and I'll predict the species!")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=X.columns)
    prediction = model.predict(input_data)[0]
    st.success(f"🌼 This flower is likely: **{prediction}**")
