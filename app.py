import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

st.title("Heart Disease Analysis App")

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

st.subheader("Raw Data")
st.dataframe(df.head())

if st.checkbox("Show Data Info"):
    buffer = []
    df.info(buf=buffer)
    st.text("\n".join(buffer))

# Feature Types
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
features_to_convert = [feature for feature in df.columns if feature not in continuous_features]
df[features_to_convert] = df[features_to_convert].astype('object')

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose Model", ["KNN", "SVM", "Decision Tree", "Random Forest"])

# Feature selection
X = df.drop("target", axis=1)
y = df["target"]

# Dummy encoding
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
if st.sidebar.button("Train and Evaluate"):
    with st.spinner("Training model..."):
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

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        st.success("Model trained!")
        st.subheader("Accuracy")
        st.write(accuracy_score(y_test, y_pred))

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

st.sidebar.markdown("---")
st.sidebar.info("Upload 'heart.csv' in the same directory to run this app.")

