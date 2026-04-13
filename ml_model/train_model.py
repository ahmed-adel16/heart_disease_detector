import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data/cleaned_data.csv")
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('Training shape:', X_train.shape)
print('Testing shape:', X_test.shape)

# Keep the search small so the notebook stays simple and fast.
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"best model: {best_model}")
predictions = best_model.predict(X_test)

metrics_df = pd.DataFrame([
    {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'best_max_depth': grid_search.best_params_['max_depth'],
        'best_min_samples_split': grid_search.best_params_['min_samples_split']
    }
])

print(metrics_df)

joblib.dump(best_model, "ml_model/decision_tree_model.joblib")
metrics_df.to_csv("reports/ml_metrics.csv", index=False)

print('Model saved to ml_model/decision_tree_model.joblib')
print('Metrics saved to reports/ml_metrics.csv')
