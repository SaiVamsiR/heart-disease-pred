import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import io

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("💓 Heart Disease Prediction App")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# Show data
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

if st.checkbox("Show Data Info"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

# Define categorical and continuous features
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
features_to_convert = [feature for feature in df.columns if feature not in continuous_features + ['target']]
df[features_to_convert] = df[features_to_convert].astype('object')

# Prepare features and target
X = df.drop("target", axis=1)
y = df["target"]
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select model
st.subheader("🤖 Choose a Machine Learning Model")
model_name = st.radio("Select Model", ["KNN", "SVM", "Decision Tree", "Random Forest"], horizontal=True)

# Define pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier())
])

if model_name == "KNN":
    pipe.set_params(clf=KNeighborsClassifier())
elif model_name == "SVM":
    pipe.set_params(clf=SVC())
elif model_name == "Decision Tree":
    pipe.set_params(clf=DecisionTreeClassifier())
elif model_name == "Random Forest":
    pipe.set_params(clf=RandomForestClassifier())

# Train model
if st.button("Train Model"):
    with st.spinner("Training..."):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        st.success("✅ Training Complete")
        st.subheader("📈 Accuracy")
        st.write(f"{accuracy_score(y_test, y_pred):.2f}")

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

# Predict on new data
st.markdown("---")
st.subheader("🔎 Predict on New Data")

input_data = {}
for col in df.columns:
    if col == "target":
        continue
    if df[col].dtype == 'object':
        input_data[col] = st.selectbox(f"{col}", sorted(df[col].unique()))
    else:
        input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

if st.button("Predict Heart Disease"):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    prediction = pipe.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ The model predicts the presence of heart disease.")
    else:
        st.success("✅ The model predicts no heart disease.")
