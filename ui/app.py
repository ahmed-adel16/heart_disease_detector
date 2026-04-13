import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from rule_based_system.expert_system import diagnose
from utils.data_preprocessing import transform_patient_data


def ml_predict(patient_data):
    transformed_patient_df = transform_patient_data(patient_data)
    model = joblib.load("ml_model/decision_tree_model.joblib")
    # get probability of heart disease
    proba = model.predict_proba(transformed_patient_df)[0][1]
    
    # binary: high or low
    if proba >= 0.5:
        return 'High heart disease risk'
    else:
        return 'Low heart disease risk'


# page setup
st.set_page_config(page_title="Heart Disease Detection", layout="wide")
st.title("Heart Disease Detection System")
st.write("Compare Expert System vs Machine Learning for heart disease risk prediction")

# sidebar with info
st.sidebar.header("About")
st.sidebar.write("This app compares:")
st.sidebar.write("- Rule-based Expert System")
st.sidebar.write("- Decision Tree ML Model")
st.sidebar.write("")
st.sidebar.write("Enter patient data to get risk assessment from both systems.")

# tabs for different sections
tab1, tab2 = st.tabs(["Prediction", "Model Dashboard"])

with tab1:
    st.header("Patient Risk Prediction")
    
    # two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Info")
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=130)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    with col2:
        st.subheader("Test Results")
        restecg = st.selectbox("Resting ECG", [0, 1, 2], 
                              format_func=lambda x: ["Normal", "ST-T abnormality", "LV hypertrophy"][x])
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise", [0, 1, 2], 
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
        ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3], 
                           format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])
    
    # predict button
    if st.button("Predict Risk", type="primary"):
        patient_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
        
        expert_result = diagnose(patient_data)
        ml_result = ml_predict(patient_data)
        
        # show results in colored boxes
        st.divider()
        st.subheader("Prediction Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.info("Expert System")
            if "High" in expert_result:
                st.error(expert_result)
            else:
                st.success(expert_result)
        
        with res_col2:
            st.info("Machine Learning Model")
            if "High" in ml_result:
                st.error(ml_result)
            else:
                st.success(ml_result)

with tab2:
    st.header("Model Performance Dashboard")
    
    # load metrics
    try:
        metrics = pd.read_csv("reports/ml_metrics.csv")
        
        # show metrics as numbers
        st.subheader("Decision Tree Model Metrics")
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric("Accuracy", f"{metrics['accuracy'].values[0]:.2%}")
        with m2:
            st.metric("Precision", f"{metrics['precision'].values[0]:.2%}")
        with m3:
            st.metric("Recall", f"{metrics['recall'].values[0]:.2%}")
        with m4:
            st.metric("F1 Score", f"{metrics['f1_score'].values[0]:.2%}")
        
        # bar chart of metrics
        st.subheader("Metrics Chart")
        chart_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [
                metrics['accuracy'].values[0],
                metrics['precision'].values[0],
                metrics['recall'].values[0],
                metrics['f1_score'].values[0]
            ]
        })
        st.bar_chart(chart_data.set_index('Metric'))
        
        # hyperparameters
        st.subheader("Best Hyperparameters")
        st.write(f"Max Depth: {metrics['best_max_depth'].values[0]}")
        st.write(f"Min Samples Split: {metrics['best_min_samples_split'].values[0]}")
        
    except:
        st.write("Metrics file not found. Please run training first.")
    
    # dataset stats
    st.divider()
    st.subheader("Dataset Statistics")
    
    try:
        df = pd.read_csv("data/cleaned_data.csv")
        
        # show data distribution
        st.write(f"Total Samples: {len(df)}")
        st.write(f"Features: {len(df.columns) - 1}")
        
        # target distribution chart
        target_counts = df['target'].value_counts()
        st.write("Target Distribution:")
        st.bar_chart(pd.DataFrame({
            'Condition': ['No Heart Disease', 'Heart Disease'],
            'Count': [target_counts.get(0, 0), target_counts.get(1, 0)]
        }).set_index('Condition'))
        
    except:
        st.write("Dataset not found.")
