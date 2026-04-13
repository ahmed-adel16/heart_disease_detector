import joblib
import pandas as pd

# load model
model = joblib.load("ml_model/decision_tree_model.joblib")


def predict_patient(patient_data):
    patient_df = pd.DataFrame([patient_data])
    # get probability of heart disease
    proba = model.predict_proba(patient_df)[0][1]
    
    # binary: high or low
    if proba >= 0.5:
        return 'High heart disease risk'
    else:
        return 'Low heart disease risk'


# get user input
def get_input():
    print("Enter patient data:")
    age = float(input("Age (scaled 0-1): "))
    sex = float(input("Sex (0=Female, 1=Male): "))
    cp = float(input("Chest pain type (0-3): "))
    trestbps = float(input("Resting blood pressure (scaled 0-1): "))
    chol = float(input("Cholesterol (scaled 0-1): "))
    fbs = float(input("Fasting blood sugar > 120 (0 or 1): "))
    restecg = float(input("Resting ECG result (0-2): "))
    thalach = float(input("Max heart rate (scaled 0-1): "))
    exang = float(input("Exercise induced angina (0 or 1): "))
    oldpeak = float(input("ST depression (scaled 0-1): "))
    slope = float(input("Slope (0-2): "))
    ca = float(input("Number of major vessels (0-4): "))
    thal = float(input("Thal (0-3): "))

    return {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }


if __name__ == "__main__":
    patient = get_input()
    result = predict_patient(patient)
    print(f"\nPrediction: {result}")