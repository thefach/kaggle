import sys
print(f"Running Python: {sys.executable}")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
print(os.getcwd())

import pandas as pd
import numpy as np

from titanic_feature_engineer import TitanicFeatureEngineer, ExtendedTitanicFeatureEngineer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

#get data
train_data = pd.read_csv("titanic/data/train.csv")
test_data = pd.read_csv("titanic/data/test.csv")

# Set up train-test split
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MLflow
tracking_uri = "file:///" + os.path.join(os.getcwd(), "titanic\mlruns")
print(tracking_uri)
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Titanic Feature Engineering Experiment")


# Define a list of transformers to evaluate
feature_transformers = {
    "TitanicFeatureEngineer": TitanicFeatureEngineer(),
    "ExtendedTitanicFeatureEngineer": ExtendedTitanicFeatureEngineer()
}

# Iterate over feature engineering transformers
for fe_name, fe_transformer in feature_transformers.items():
    # MLflow tracking context
    with mlflow.start_run(run_name=fe_name):
        
        # Step 1: Apply Feature Engineering
        X_train_fe = fe_transformer.fit_transform(X_train)
        X_test_fe = fe_transformer.transform(X_test)

        # Step 2: Dynamically Determine Preprocessing Columns
        # Separate numerical and categorical columns after feature engineering
        num_features = X_train_fe.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X_train_fe.select_dtypes(include=['object']).columns

        # Step 3: Define Preprocessing and Model Pipeline with Imputer
        preprocessor = ColumnTransformer([
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values for numerical features
                ('scaler', StandardScaler())  # Scale numerical features
            ]), num_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values for categorical features
                ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
            ]), cat_features)
        ])
        
        pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # Train the model
        pipeline.fit(X_train_fe, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test_fe)

        # Step 4: Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Step 5: Prepare Input Example and Signature for MLflow Logging
        input_example = X_train_fe.iloc[:5]  # Example of input for the model (using first 5 rows)
        signature = infer_signature(X_train_fe, y_pred)

        # Step 6: Log Parameters, Metrics, and Model to MLflow
        mlflow.log_param("Feature Engineer", fe_name)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1 Score", f1)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=f"RandomForest_{fe_name}",
            signature=signature,
            input_example=input_example
        )

        print(f"Experiment {fe_name} completed successfully: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

mlflow.end_run()
