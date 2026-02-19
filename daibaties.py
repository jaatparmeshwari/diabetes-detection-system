# diabetes_app.py

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("Patient_Data.csv")

df = load_data()
st.title("🩺 Diabetes Prediction App")

# Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Models
log_model = LogisticRegression()
svm_model = SVC(probability=True)

log_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Evaluation
log_prec = precision_score(y_test, log_model.predict(X_test))
log_recall = recall_score(y_test, log_model.predict(X_test))

svm_prec = precision_score(y_test, svm_model.predict(X_test))
svm_recall = recall_score(y_test, svm_model.predict(X_test))

st.sidebar.subheader("📊 Model Metrics")
st.sidebar.write(f"**Logistic Regression** - Precision: {log_prec:.2f}, Recall: {log_recall:.2f}")
st.sidebar.write(f"**SVM** - Precision: {svm_prec:.2f}, Recall: {svm_recall:.2f}")

# Input Fields
st.header("🔬 Enter Patient Medical Info")

features = {}
for col in df.columns[:-1]:
    val = st.number_input(col, min_value=0.0, value=float(df[col].mean()))
    features[col] = val

model_choice = st.selectbox("Choose Model", ["Logistic Regression", "SVM"])

# Predict
input_array = np.array(list(features.values())).reshape(1, -1)
input_scaled = scaler.transform(input_array)

if st.button("Predict Diabetes Risk"):
    if model_choice == "Logistic Regression":
        pred = log_model.predict(input_scaled)[0]
        prob = log_model.predict_proba(input_scaled)[0][1]
    else:
        pred = svm_model.predict(input_scaled)[0]
        prob = svm_model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ High Risk of Diabetes ({prob*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({(1-prob)*100:.2f}%)")
