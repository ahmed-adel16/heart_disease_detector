from rule_based_system.rules import RISK_RULES


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
