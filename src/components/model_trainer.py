# Import necessary modules
import sys
import os
import numpy as np
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

# Define a configuration class for model trainer settings
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

# Define the main class responsible for model training
class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train and evaluate multiple models to find the best performing one based on a weighted scoring system.
        
        Parameters:
        train_array (array-like): Training data array with features and target.
        test_array (array-like): Testing data array with features and target.

        Returns:
        tuple: A tuple containing the R2 score, MAE, MSE, RMSE of the best model on test data, and the name of the best model.
        """

        try:
            logging.info("Splitting Training and Testing Input Data.")
            
            # Split the train and test arrays into features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Initialize models
            models = {
                'Random Forest': RandomForestRegressor(random_state=6112024),
                'Gradient Boosting': GradientBoostingRegressor(random_state=6112024),
                'XGBoost': XGBRegressor(random_state=6112024),
                'AdaBoost': AdaBoostRegressor(random_state=6112024),
                'Bagging': BaggingRegressor(random_state=6112024),
                'Kernel Ridge': KernelRidge(),
                'Hist Gradient Boosting': HistGradientBoostingRegressor(random_state=6112024)
            }

            # Define parameter grids for hyperparameter tuning
            param_grids = {
                'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
                'Gradient Boosting': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5]},
                'XGBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5]},
                'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]},
                'Bagging': {'n_estimators': [10, 50, 100], 'max_samples': [0.5, 0.75, 1.0], 'max_features': [0.5, 0.75, 1.0]},
                'Kernel Ridge': {'alpha': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': [0.01, 0.1, 1.0, None]},
                'Hist Gradient Boosting': {'learning_rate': [0.01, 0.1, 0.2], 'max_iter': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_leaf': [20, 50, 100]}
            }

            # Evaluate models and get a report of their performance
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=param_grids)

            # Define weights for different metrics
            weights = {
                'R2 Score': 0.4,
                'Mean Absolute Error': 0.2,
                'Mean Squared Error': 0.2,
                'Root Mean Squared Error': 0.2
            }

            # Calculate weighted scores for each model
            weighted_scores = {}
            for model_name, metrics in model_report.items():
                test_metrics = metrics["Test Metrics"]
                weighted_score = (
                    weights['R2 Score'] * test_metrics["R2 Score"] -
                    weights['Mean Absolute Error'] * test_metrics["Mean Absolute Error"] -
                    weights['Mean Squared Error'] * test_metrics["Mean Squared Error"] -
                    weights['Root Mean Squared Error'] * test_metrics["Root Mean Squared Error"]
                )
                weighted_scores[model_name] = weighted_score

            # Find the best model based on the weighted score
            best_model_name = max(weighted_scores, key=weighted_scores.get)
            best_model = models[best_model_name]

            logging.info("Best Model found based on weighted scoring system.")
            logging.info(f"Report on All Models: {model_report}")
            
            # Fit the best model
            best_model.fit(X_train, y_train)
            
            # Save the best model to a file
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Predict on test data and calculate various evaluation metrics
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)
            rmse = np.sqrt(mse)

            return (r2, mae, mse, rmse, best_model_name)

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)
