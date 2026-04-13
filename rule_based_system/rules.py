RISK_RULES = {
    # existing rules
    ("older_age", "high_cholesterol", "high_blood_pressure"): "High heart disease risk",
    ("older_age", "high_blood_pressure", "exercise_angina"): "High heart disease risk",
    ("high_cholesterol", "exercise_angina", "low_max_heart_rate"): "High heart disease risk",
    ("high_st_depression", "exercise_angina"): "High heart disease risk",
    ("many_blocked_vessels", "abnormal_thal"): "High heart disease risk",
    ("chest_pain_problem", "exercise_angina", "high_st_depression"): "High heart disease risk",
    ("high_fasting_blood_sugar", "high_blood_pressure", "older_age"): "High heart disease risk",
    ("abnormal_ecg", "low_max_heart_rate"): "High heart disease risk",
    ("chest_pain_problem", "high_cholesterol"): "High heart disease risk",
    ("older_age", "high_cholesterol"): "High heart disease risk",
    # 10 new simpler rules
    ("exercise_angina",): "High heart disease risk",
    ("many_blocked_vessels",): "High heart disease risk",
    ("older_age", "exercise_angina"): "High heart disease risk",
    ("high_cholesterol", "high_blood_pressure"): "High heart disease risk",
    ("chest_pain_problem", "exercise_angina"): "High heart disease risk",
    ("abnormal_thal",): "High heart disease risk",
    ("high_st_depression", "chest_pain_problem"): "High heart disease risk",
    ("older_age", "many_blocked_vessels"): "High heart disease risk",
    ("high_fasting_blood_sugar", "exercise_angina"): "High heart disease risk",
    ("abnormal_ecg",): "High heart disease risk",
    # low risk rules
    ("young_age", "normal_blood_pressure", "normal_cholesterol", "good_max_heart_rate"): "Low heart disease risk",
    ("young_age", "no_exercise_angina", "low_st_depression"): "Low heart disease risk",
}
