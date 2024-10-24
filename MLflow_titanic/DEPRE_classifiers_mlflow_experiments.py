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

#get data
train_data = pd.read_csv("titanic/data/train.csv").drop(columns=['PassengerId'])
test_data = pd.read_csv("titanic/data/test.csv").drop(columns=['PassengerId'])

# Set up train-test split
X = train_data.drop(columns=['Survived'])
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MLflow
tracking_uri = "file:///" + os.path.join(os.getcwd(), "titanic\mlruns")
print(tracking_uri)
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Titanic Feature Engineering Experiment")

# Define a dictionary of custom transformers for FE to be evaluated (can be extended, calls titanic_feature_engineer.py)
feature_transformers = {
    "TitanicFeatureEngineer": TitanicFeatureEngineer(),
    "ExtendedTitanicFeatureEngineer": ExtendedTitanicFeatureEngineer()
}

########## Logistic Regression Classifier ##########
# Define parameter grid for Logistic Regression
lr_param_grid = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__penalty': ['l2'],
    'model__solver': ['lbfgs', 'liblinear']
}

for fe_name, fe_transformer in feature_transformers.items():
    lr_fe_name = f"{fe_name}_LogisticRegression"
    with mlflow.start_run(run_name=lr_fe_name):
        
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
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Fill missing values for numerical features
                ('scaler', StandardScaler())  # Scale numerical features
            ]), num_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values for categorical features
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encode categorical features as dense matrix
            ]), cat_features)
        ])
        
        pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])

        # Step 4: Perform Grid Search to Fine-Tune Logistic Regression
        grid_search = GridSearchCV(estimator=pipeline, param_grid=lr_param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_fe, y_train)

        # Get the best model from Grid Search
        best_pipeline = grid_search.best_estimator_

        # Make predictions
        y_pred = best_pipeline.predict(X_test_fe)

        # Step 5: Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Step 6: Prepare Input Example and Signature for MLflow Logging
        input_example = X_train_fe.iloc[:5]  # Example of input for the model (using first 5 rows)
        signature = infer_signature(X_train_fe, y_pred)

        # Step 7: Log Parameters, Metrics, and Model to MLflow
        mlflow.log_param("Feature Engineer", fe_name)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1 Score", f1)
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path=f"LogisticRegression_{fe_name}",
            signature=signature,
            input_example=input_example
        )

        print(f"Experiment {lr_fe_name} completed successfully: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

########## Support Vector Classifier (SVC) ##########
# Define parameter grid for SVC
svc_param_grid = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear', 'rbf', 'poly'],
    'model__gamma': ['scale', 'auto']
}

for fe_name, fe_transformer in feature_transformers.items():
    svc_fe_name = f"{fe_name}_SVC"
    with mlflow.start_run(run_name=svc_fe_name):
        
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
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Fill missing values for numerical features
                ('scaler', StandardScaler())  # Scale numerical features
            ]), num_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values for categorical features
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encode categorical features as dense matrix
            ]), cat_features)
        ])
        
        pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('model', SVC(random_state=42))
        ])

        # Step 4: Perform Grid Search to Fine-Tune SVC
        grid_search = GridSearchCV(estimator=pipeline, param_grid=svc_param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_fe, y_train)

        # Get the best model from Grid Search
        best_pipeline = grid_search.best_estimator_

        # Make predictions
        y_pred = best_pipeline.predict(X_test_fe)

        # Step 5: Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Step 6: Prepare Input Example and Signature for MLflow Logging
        input_example = X_train_fe.iloc[:5]  # Example of input for the model (using first 5 rows)
        signature = infer_signature(X_train_fe, y_pred)

        # Step 7: Log Parameters, Metrics, and Model to MLflow
        mlflow.log_param("Feature Engineer", fe_name)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1 Score", f1)
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path=f"SVC_{fe_name}",
            signature=signature,
            input_example=input_example
        )

        print(f"Experiment {svc_fe_name} completed successfully: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

########## Random Forest Classifier ##########
# Define parameter grid for Random Forest
gr_param_grid = {
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_depth': [None, 10, 20, 50],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4, 5, 10]
}

# Iterate over feature engineering transformers
for fe_name, fe_transformer in feature_transformers.items():
    rf_fe_name = f"{fe_name}_RFClassifier"  
    # MLflow tracking context
    with mlflow.start_run(run_name=rf_fe_name):
        
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
                ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Fill missing values for numerical features
                ('scaler', StandardScaler())  # Scale numerical features
            ]), num_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values for categorical features
                ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
            ]), cat_features)
        ])
        
        pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('model', RandomForestClassifier(random_state=42))
        ])

        # Step 4: Perform Grid Search to Fine-Tune Random Forest
        grid_search = GridSearchCV(estimator=pipeline, param_grid=gr_param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_fe, y_train)

        # Get the best model from Grid Search
        best_pipeline = grid_search.best_estimator_

        # Make predictions
        y_pred = best_pipeline.predict(X_test_fe)

        # Step 5: Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Step 6: Prepare Input Example and Signature for MLflow Logging
        input_example = X_train_fe.iloc[:5]  # Example of input for the model (using first 5 rows)
        signature = infer_signature(X_train_fe, y_pred)

        # Step 7: Log Parameters, Metrics, and Model to MLflow
        mlflow.log_param("Feature Engineer", fe_name)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1 Score", f1)
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path=f"RandomForest_{fe_name}",
            signature=signature,
            input_example=input_example
        )

        print(f"Experiment {fe_name} completed successfully: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

########## Histogram-based Gradient Boosting Classification Tree ##########
# Define parameter grid for HistGBClassifier
gb_param_grid = {
    'model__max_iter': [100, 200, 300],
    'model__max_depth': [3, 5, 10],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__l2_regularization': [0, 0.1, 1.0]
}

for fe_name, fe_transformer in feature_transformers.items():
    gb_fe_name = f"{fe_name}_HistGBClassifier"
    with mlflow.start_run(run_name=gb_fe_name):
        
        # Step 1: Apply Feature Engineering
        X_train_fe = fe_transformer.fit_transform(X_train)
        X_test_fe = fe_transformer.transform(X_test)

        # Step 2: Dynamically Determine Preprocessing Columns
        # Separate numerical and categorical columns after feature engineering
        num_features = X_train_fe.select_dtypes(include=['int64', 'float64']).columns
        cat_features = X_train_fe.select_dtypes(include=['object']).columns

        # Step 3: Define Preprocessing and Model Pipeline (HistGradientBoosting handles missing values)
        preprocessor = ColumnTransformer([
            ('num', Pipeline(steps=[
                ('scaler', StandardScaler())  # Scale numerical features
            ]), num_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values for categorical features
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encode categorical features as dense matrix
            ]), cat_features)
        ])
        
        pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('model', HistGradientBoostingClassifier(random_state=42))
        ])

        # Step 4: Perform Grid Search to Fine-Tune HistGradientBoosting
        grid_search = GridSearchCV(estimator=pipeline, param_grid=gb_param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_fe, y_train)

        # Get the best model from Grid Search
        best_pipeline = grid_search.best_estimator_

        # Make predictions
        y_pred = best_pipeline.predict(X_test_fe)

        # Step 5: Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Step 6: Prepare Input Example and Signature for MLflow Logging
        input_example = X_train_fe.iloc[:5]  # Example of input for the model (using first 5 rows)
        signature = infer_signature(X_train_fe, y_pred)

        # Step 7: Log Parameters, Metrics, and Model to MLflow
        mlflow.log_param("Feature Engineer", fe_name)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1 Score", f1)
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path=f"HistGBClassifier_{fe_name}",
            signature=signature,
            input_example=input_example
        )

        print(f"Experiment {gb_fe_name} completed successfully: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

mlflow.end_run() # End the experiment
