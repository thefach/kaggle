import sys
print(f"Running Python: {sys.executable}")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
print(os.getcwd())

import pandas as pd

from titanic_feature_engineer import TitanicFeatureEngineer, ExtendedTitanicFeatureEngineer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def run_model_experiment(model_name, model, param_grid, fe_name, fe_transformer, X_train, X_test, y_train, y_test):
    """
    Runs model training, evaluation, and logging to MLflow for a given model and feature transformer.

    Parameters:
        model_name (str): Name of the model.
        model: Machine learning model instance.
        param_grid (dict): GridSearchCV parameter grid.
        fe_name (str): Name of the feature transformer.
        fe_transformer: Feature transformer instance.
        X_train, X_test, y_train, y_test: Training and test data.
    """
    # Start an MLflow run
    run_name = f"{fe_name}_{model_name}"
    with mlflow.start_run(run_name=run_name):
        # Step 1: Apply Feature Engineering
        X_train_fe = fe_transformer.fit_transform(X_train)
        X_test_fe = fe_transformer.transform(X_test)

        # Step 2: Dynamically Determine Preprocessing Columns
        num_features = X_train_fe.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X_train_fe.select_dtypes(include=['object']).columns

        # Step 3: Define Preprocessing and Model Pipeline
        # Handle missing values differently if model supports it natively
        if isinstance(model, HistGradientBoostingClassifier):
            # HistGradientBoostingClassifier can handle missing values natively
            preprocessor = ColumnTransformer([
                ('num', Pipeline(steps=[
                    ('scaler', StandardScaler())  # Scale numerical features
                ]), num_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute categorical features
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode categorical features
                ]), cat_features)
            ])
        else:
            # Other models require imputation for numerical features
            preprocessor = ColumnTransformer([
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Impute numerical features
                    ('scaler', StandardScaler())  # Scale numerical features
                ]), num_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute categorical features
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode categorical features
                ]), cat_features)
            ])
        
        # Step 4: Create Pipeline (Preprocessing + Model)
        pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('model', model)
        ])

        # Step 5: Perform Grid Search with Cross-Validation
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_fe, y_train)

        # Get the best model
        best_pipeline = grid_search.best_estimator_

        # Step 6: Make Predictions on Test Set
        y_pred = best_pipeline.predict(X_test_fe)

        # Step 7: Calculate Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Step 8: Prepare Input Example and Signature for MLflow Logging
        input_example = X_train_fe.iloc[:5]
        signature = infer_signature(X_train_fe, y_pred)

        # Step 9: Log Parameters, Metrics, and Model to MLflow
        mlflow.log_param("Feature Engineer", fe_name)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1 Score", f1)
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path=f"{model_name}_{fe_name}",
            signature=signature,
            input_example=input_example
        )

        print(f"Experiment {run_name} completed successfully: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

if __name__ == '__main__':
    # Load data
    train_data = pd.read_csv("MLflow_titanic/data/train.csv").drop(columns=['PassengerId'])
    test_data = pd.read_csv("MLflow_titanic/data/test.csv").drop(columns=['PassengerId'])

    # Set up train-test split
    X = train_data.drop(columns=['Survived'])
    y = train_data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize MLflow
    tracking_uri = "file:///" + os.path.join(os.getcwd(), "MLflow_titanic\mlruns")
    print(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Titanic Feature Engineering Experiment")

    # Define feature transformers
    feature_transformers = {
        "TitanicFeatureEngineer": TitanicFeatureEngineer(),
        "ExtendedTitanicFeatureEngineer": ExtendedTitanicFeatureEngineer()
    }

    # Define models and their parameter grids
    models = [
        # Logistic Regression
        {
            'model_name': 'LogisticRegression',
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'param_grid': {
                'model__C': [0.01, 0.1, 1, 10],
                'model__penalty': ['l2'],
                'model__solver': ['lbfgs', 'liblinear']
            }
        },
        # Support Vector Classifier
        {
            'model_name': 'SVC',
            'model': SVC(random_state=42),
            'param_grid': {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['linear', 'rbf', 'poly'],
                'model__gamma': ['scale', 'auto']
            }
        },
        # Random Forest Classifier
        {
            'model_name': 'RandomForestClassifier',
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'model__n_estimators': [50, 100, 200, 300],
                'model__max_depth': [None, 10, 20, 50],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4, 5, 10]
            }
        },
        # Histogram-based Gradient Boosting Classifier
        {
            'model_name': 'HistGradientBoostingClassifier',
            'model': HistGradientBoostingClassifier(random_state=42),
            'param_grid': {
                'model__max_iter': [100, 200, 300],
                'model__max_depth': [3, 5, 10],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__l2_regularization': [0, 0.1, 1.0]
            }
        }
    ]

    # Run experiments for each model and feature transformer combination
    for model_info in models:
        for fe_name, fe_transformer in feature_transformers.items():
            run_model_experiment(
                model_name=model_info['model_name'],
                model=model_info['model'],
                param_grid=model_info['param_grid'],
                fe_name=fe_name,
                fe_transformer=fe_transformer,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )

    # End MLflow run
    mlflow.end_run()
