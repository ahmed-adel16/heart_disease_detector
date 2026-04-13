import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

CATEGORICAL_COLS = ['cp', 'restecg', 'slope', 'thal']
NUMERICAL_COLS = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca']


def transform_patient_data(patient_data):
    patient_df = pd.DataFrame([patient_data])
    preprocessor = joblib.load("utils/heart_preprocessor.joblib")
    scaler = preprocessor['scaler']

    df_encoded = pd.get_dummies(patient_df, columns=CATEGORICAL_COLS, drop_first=True)

    # add missing columns with 0
    for col in preprocessor['columns']:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[preprocessor['columns']]

    # scale numerical columns
    df_encoded[NUMERICAL_COLS] = scaler.transform(df_encoded[NUMERICAL_COLS])

    return df_encoded


# main preprocessing code (runs when file is executed directly)
df = pd.read_csv("data/heart.csv")
df = df.fillna(df.median())

y = df['target']
df_features = df.drop('target', axis=1)

# one-hot encoding: convert categories to binary columns
df_encoded = pd.get_dummies(df_features, columns=CATEGORICAL_COLS, drop_first=True)
column_names = df_encoded.columns.tolist()

# scale numerical columns using sklearn
scaler = MinMaxScaler()
df_encoded[NUMERICAL_COLS] = scaler.fit_transform(df_encoded[NUMERICAL_COLS])

df_processed = df_encoded.copy()
df_processed['target'] = y.values

preprocessor = {'columns': column_names, 'scaler': scaler}
joblib.dump(preprocessor, "utils/heart_preprocessor.joblib")
df_processed.to_csv("data/cleaned_data.csv", index=False)

print('cleaned_data.csv saved successfully.')
print(f'Feature columns: {len(column_names)}')
