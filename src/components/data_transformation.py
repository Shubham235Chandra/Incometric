# Import necessary modules
import sys
import os

import numpy as np 
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Define a configuration class for data transformation settings
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

# Define the main class responsible for data transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Function to create a preprocessing object
    def get_data_transformer_object(self):
        '''This function is Responsible for Data Transformation'''
        
        try:
            # Define the categorical and numerical columns
            categorical_columns = ["Primary_Mode_of_Transportation", "Education_Level", "Occupation", "Marital_Status", "Living_Standards", "Gender", "Homeownership_Status", "Location", "Type_of_Housing", "Employment_Status"]
            numerical_columns = ["Work_Experience", "Number_of_Dependents", "Household_Size", "Age", "Cluster"]

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Create pipelines for numerical and categorical data processing
            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
            cat_pipeline = Pipeline(steps=[("one_hot_encoder", OneHotEncoder()), ("scaler", StandardScaler(with_mean=False))])

            # Combine the pipelines into a single preprocessor
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)

    # Function to initiate data transformation on train and test datasets
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read the train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read Train and Test Data Completed")
            
            # Obtain the preprocessing object
            logging.info("Obtaining Preprocessing Object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target column and numerical columns
            target_column_name = "Income"
            numerical_columns = ["Work_Experience", "Number_of_Dependents", "Household_Size", "Age"]

            # Separate input features and target feature from the train and test data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying Preprocessing Object on Training Dataframe and Testing Dataframe.")

            # Apply the preprocessing object to the train and test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the transformed input features with the target feature
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved Preprocessing Object.")

            # Save the preprocessing object for future use
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path,)

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)
