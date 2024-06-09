import os
import sys
import pickle
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import GridSearchCV
import numpy as np

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    Parameters:
    file_path (str): The file path where the object should be saved.
    obj (object): The object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models using GridSearchCV and return their performance on the test set.
    
    Parameters:
    X_train (array-like): Training feature data.
    y_train (array-like): Training target data.
    X_test (array-like): Testing feature data.
    y_test (array-like): Testing target data.
    models (dict): Dictionary of models to evaluate.
    param (dict): Dictionary of parameter grids for each model.

    Returns:
    dict: A dictionary containing the evaluation metrics for each model.
    """
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = param[model_name]

            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_metrics = {
                "R2 Score": r2_score(y_train, y_train_pred),
                "Mean Absolute Error": mean_absolute_error(y_train, y_train_pred),
                "Mean Squared Error": mean_squared_error(y_train, y_train_pred),
                "Root Mean Squared Error": np.sqrt(mean_squared_error(y_train, y_train_pred))
            }

            test_metrics = {
                "R2 Score": r2_score(y_test, y_test_pred),
                "Mean Absolute Error": mean_absolute_error(y_test, y_test_pred),
                "Mean Squared Error": mean_squared_error(y_test, y_test_pred),
                "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, y_test_pred))
            }

            if len(np.unique(y_train)) <= 2:  # Check if the problem is binary classification
                train_metrics.update({
                    "Accuracy": accuracy_score(y_train, y_train_pred),
                    "Precision": precision_score(y_train, y_train_pred, zero_division=1),
                    "Recall": recall_score(y_train, y_train_pred, zero_division=1),
                    "F1 Score": f1_score(y_train, y_train_pred, zero_division=1)
                })

                test_metrics.update({
                    "Accuracy": accuracy_score(y_test, y_test_pred),
                    "Precision": precision_score(y_test, y_test_pred, zero_division=1),
                    "Recall": recall_score(y_test, y_test_pred, zero_division=1),
                    "F1 Score": f1_score(y_test, y_test_pred, zero_division=1)
                })

            report[model_name] = {
                "Train Metrics": train_metrics,
                "Test Metrics": test_metrics
            }

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file using pickle.
    
    Parameters:
    file_path (str): The file path from which the object should be loaded.

    Returns:
    object: The loaded object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
