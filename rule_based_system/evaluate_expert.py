import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rule_based_system.expert_system import diagnose


def expert_predict(patient_data):
    result = diagnose(patient_data)
    return 1 if "High" in result else 0


df = pd.read_csv("data/heart.csv")

# use same test split as ml model for fair comparison
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# convert test data back to dict for expert system
predictions = []
for _, row in X_test.iterrows():
    patient_dict = row.to_dict()
    pred = expert_predict(patient_dict)
    predictions.append(pred)

# calculate metrics
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

# save metrics
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("reports/expert_metrics.csv", index=False)
print("\nMetrics saved to reports/expert_metrics.csv")
