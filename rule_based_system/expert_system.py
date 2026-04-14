import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


from rules import RISK_RULES


def extract_facts(patient_data):
    facts = []

    if patient_data["age"] >= 55:
        facts.append("older_age")
    if patient_data["age"] < 45:
        facts.append("young_age")

    if patient_data["trestbps"] >= 140:
        facts.append("high_blood_pressure")
    else:
        facts.append("normal_blood_pressure")

    if patient_data["chol"] >= 240:
        facts.append("high_cholesterol")
    else:
        facts.append("normal_cholesterol")

    if patient_data["fbs"] == 1:
        facts.append("high_fasting_blood_sugar")
    if patient_data["restecg"] != 0:
        facts.append("abnormal_ecg")
    if patient_data["thalach"] < 130:
        facts.append("low_max_heart_rate")
    if patient_data["thalach"] >= 150:
        facts.append("good_max_heart_rate")
    if patient_data["exang"] == 1:
        facts.append("exercise_angina")
    else:
        facts.append("no_exercise_angina")
    if patient_data["oldpeak"] >= 2:
        facts.append("high_st_depression")
    else:
        facts.append("low_st_depression")
    if patient_data["cp"] in [1, 2, 3]:
        facts.append("chest_pain_problem")
    if patient_data["ca"] >= 2:
        facts.append("many_blocked_vessels")
    if patient_data["thal"] in [1, 3]:
        facts.append("abnormal_thal")

    return facts


def diagnose(patient_data):
    patient_facts = extract_facts(patient_data)

    for condition, diagnosis in RISK_RULES.items():
        if set(condition).issubset(set(patient_facts)):
            return diagnosis

    return "Low heart disease risk"


def evaluate_expert_system():
    def expert_predict(patient_data):
        result = diagnose(patient_data)
        return 1 if "High" in result else 0

    df = pd.read_csv("data/heart.csv")
    df = df.fillna(df.median(numeric_only=True))

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    predictions = []
    for _, row in X_test.iterrows():
        patient_dict = row.to_dict()
        pred = expert_predict(patient_dict)
        predictions.append(pred)

    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions)
    }

    print("Expert System Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.6f}")
    print(f"  Precision: {metrics['precision']:.6f}")
    print(f"  Recall:    {metrics['recall']:.6f}")
    print(f"  F1-score:  {metrics['f1_score']:.6f}")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("reports/expert_metrics.csv", index=False)
    print("\nMetrics saved to reports/expert_metrics.csv")
    return metrics


if __name__ == "__main__":
    evaluate_expert_system()
