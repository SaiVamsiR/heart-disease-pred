# 💓 Heart Disease Prediction App

This is an interactive **Streamlit web application** that predicts the presence of heart disease in a patient based on medical features. The app is built using Python and machine learning techniques.

---

## 🚀 Features

- 📊 **Explore dataset**: View patient health data and get detailed information on each feature.
- 🧠 **Choose your model**: Train and evaluate different ML models like KNN, SVM, Decision Tree, and Random Forest.
- ✍️ **Enter your own data**: Input patient information to get a real-time heart disease prediction.
- ✅ **User-friendly UI**: No coding needed. Everything is point-and-click. 

---

## 📦 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn (ML models)
- Streamlit (interactive frontend)

---

## 📂 Dataset Overview

The app uses the **Heart Disease Dataset**

Each row represents a patient and contains 13 features + 1 target value.

### Key Features:
- **age**: Age in years
- **sex**: Gender (0 = female, 1 = male)
- **cp**: Chest pain type (0 = typical angina, ..., 3 = asymptomatic)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: ECG results (0 = normal, 1 = abnormal, 2 = hypertrophy)
- **thalach**: Max heart rate achieved
- **exang**: Exercise-induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression relative to rest
- **slope**: Slope of ST segment (0 = up, 1 = flat, 2 = down)
- **ca**: Number of major vessels (0-3)
- **thal**: Thalassemia type (1 = normal, 2 = fixed defect, 3 = reversible defect)

### Target:
- `0` → No heart disease  
- `1` → Heart disease present

---


## 🧠 Model Training Process

The project trains and evaluates multiple machine learning models:

1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical features
   - Scaling numeric features

2. **Model Selection**
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Decision Tree
   - Random Forest

3. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
   - ROC Curve

---

## 🎮 Interactive Streamlit App

A simple user interface is provided using Streamlit where users can:

- 👁️ View the dataset
- ⚙️ Train a model with a click
- 📝 Input new patient data
- 🩺 Predict heart disease likelihood instantly

