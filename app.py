import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import io

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Ignore warnings
warnings.filterwarnings("ignore")

# Page title
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")
st.title("💓 Heart Disease Analysis and Prediction")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# Display raw data
st.subheader("🔍 Preview Data")
st.dataframe(df.head())

# Show info
if st.checkbox("Show Data Info"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

# Feature categorization
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
features_to_convert = [feature for feature in df.columns if feature not in continuous_features + ['target']]
df[features_to_convert] = df[features_to_convert].astype('object')

# Sidebar - model selection
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", ["KNN", "SVM", "Decision Tree", "Random Forest"])

# Prepare features and target
X = df.drop("target", axis=1)
y = df["target"]
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
if st.sidebar.button("Train and Evaluate"):
    st.subheader("🚀 Training Model")
    with st.spinner("Training in progress..."):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())  # default
        ])

        # Set model
        if model_name == "KNN":
            pipe.set_params(clf=KNeighborsClassifier())
        elif model_name == "SVM":
            pipe.set_params(clf=SVC())
        elif model_name == "Decision Tree":
            pipe.set_params(clf=DecisionTreeClassifier())
        elif model_name == "Random Forest":
            pipe.set_params(clf=RandomForestClassifier())

        # Fit and predict
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Output
        st.success("✅ Model training completed!")
        st.subheader("📈 Accuracy")
        st.write(f"{accuracy_score(y_test, y_pred):.2f}")

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.info("Ensure `heart.csv` is in the same folder to run the app.")
