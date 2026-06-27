%%writefile app.py

import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.title("HLPI Risk Prediction System")

# Dummy training data (replace later with real model)
X = np.random.rand(100, 3) * 100
y = (X[:,1] > 50).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

p = st.slider("Performance Score", 0, 100, 50)
h = st.slider("Health Risk", 0, 100, 50)
l = st.slider("Life Score", 0, 100, 50)

if st.button("Predict"):
    inp = scaler.transform([[p, h, l]])
    pred = model.predict(inp)

    if pred[0] == 1:
        st.error("High Risk 🔴")
    else:
        st.success("Low Risk 🟢")
        