import pickle
import pandas as pd
import sys
import os

from titanic_feature_engineer import TitanicFeatureEngineer
from sklearn.pipeline import Pipeline

# Path to the saved model pickle (replace with the provided path)
model_path = "mlruns/370133384786671548/72bd08a1acb244dbb3ad530e54ee43de/artifacts/RandomForestClassifier_TitanicFeatureEngineer/model.pkl"

# Load the model
try:
    with open(model_path, 'rb') as file:
        best_model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file at {model_path} was not found.")
    sys.exit(1)

# Load the test data
test_data_path = "data/test.csv"
if not os.path.exists(test_data_path):
    print(f"Error: The test data file at {test_data_path} was not found.")
    sys.exit(1)

test_data = pd.read_csv(test_data_path)

# Apply feature engineering
feature_engineer = TitanicFeatureEngineer()
test_data_features = feature_engineer.transform(test_data)

# Prepare the test data (exclude PassengerId for prediction)
passenger_ids = test_data['PassengerId']
test_data_features = test_data_features.drop(columns=['PassengerId'])

# Apply the preprocessing pipeline from the trained model to the test data
try:
    # Extract preprocessing pipeline from the loaded model
    preprocessing_pipeline = best_model.named_steps['preprocessing']
    test_data_preprocessed = preprocessing_pipeline.transform(test_data_features)
    print("Test data preprocessed successfully.")
except KeyError:
    print("Error: The loaded model does not contain a preprocessing step.")
    sys.exit(1)
except ValueError as e:
    print(f"Error during preprocessing: {e}")
    sys.exit(1)

# Generate predictions
try:
    predictions = best_model.named_steps['model'].predict(test_data_preprocessed)
    print("Predictions generated successfully.")
except AttributeError:
    print("Error: The loaded model does not have a predict method.")
    sys.exit(1)

# Prepare the submission DataFrame
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})

# Save the submission file
submission_path = "submission/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved successfully at {submission_path}.")
