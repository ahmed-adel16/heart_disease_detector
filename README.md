# Heart Disease Detection

This project follows the PDF requirements for a simple heart disease detection system using:

- A rule-based expert system
- A decision tree machine learning model
- Data preprocessing and visualization
- A simple Streamlit interface
- One shared preprocessor to reduce redundancy

## Project Structure

- `data/` contains the raw and cleaned dataset
- `utils/` contains the preprocessing notebook
- `notebooks/` contains analysis and summary notebooks
- `rule_based_system/` contains the expert system notebooks
- `ml_model/` contains model training and prediction notebooks
- `reports/` contains the comparison report
- `ui/` contains the Streamlit app

## How To Run

1. Run `utils/data_preprocessing.ipynb` to clean and preprocess the data
2. Run `notebooks/data_analysis.ipynb` for data visualization
3. Run `ml_model/train_model.py` to train the decision tree model
4. Run `ml_model/predict.py` to test predictions
5. Run `rule_based_system/rules.py` (contains the risk rules)
6. Run `rule_based_system/expert_system.py` to test the expert system
7. Start the UI with `streamlit run ui/app.py`

## Shared Preprocessor

- `utils/heart_preprocessor.joblib` is the saved preprocessor used by the prediction code and the Streamlit UI
- `utils/preprocessing.py` contains the `transform_patient_data` function for scaling new patient data
