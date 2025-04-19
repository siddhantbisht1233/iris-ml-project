
import streamlit as st
import joblib
import numpy as np

model = joblib.load('iris_model.pkl')

st.title("🌸 Iris Flower Predictor")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    classes = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    st.success(f"Predicted Iris: {classes[prediction]}")
