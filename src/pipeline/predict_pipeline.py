import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.pipeline.kmeans_clustering import predict_cluster

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()

    def load_model(self):
        try:
            return load_object(file_path=self.model_path)
        except Exception as e:
            raise CustomException(e, sys)

    def load_preprocessor(self):
        try:
            return load_object(file_path=self.preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys)

    def predicts(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

    def add_clusters(self, age, work_experience, household_size, living_standards):
        try:
            predict = predict_cluster(age, work_experience, household_size, living_standards)
            return predict
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, primary_mode_of_transportation: str, education_level: str, occupation: str, marital_status: str, living_standards: str, gender: str,
                 homeownership_status: str, location: str, type_of_housing: str, employment_status: str, work_experience: int, number_of_dependents: int, household_size: int, age: int):
        self.primary_mode_of_transportation = primary_mode_of_transportation
        self.education_level = education_level
        self.occupation = occupation
        self.marital_status = marital_status
        self.living_standards = living_standards
        self.gender = gender
        self.homeownership_status = homeownership_status
        self.location = location
        self.type_of_housing = type_of_housing
        self.employment_status = employment_status
        self.work_experience = work_experience
        self.number_of_dependents = number_of_dependents
        self.household_size = household_size
        self.age = age

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Primary_Mode_of_Transportation": [self.primary_mode_of_transportation],
                "Education_Level": [self.education_level],
                "Occupation": [self.occupation],
                "Marital_Status": [self.marital_status],
                "Living_Standards": [self.living_standards],
                "Gender": [self.gender],
                "Homeownership_Status": [self.homeownership_status],
                "Location": [self.location],
                "Type_of_Housing": [self.type_of_housing],
                "Employment_Status": [self.employment_status],
                "Work_Experience": [self.work_experience],
                "Number_of_Dependents": [self.number_of_dependents],
                "Household_Size": [self.household_size],
                "Age": [self.age],
            }

            data_df = pd.DataFrame(custom_data_input_dict)
            predict_pipeline = PredictPipeline()
            #print("Data Input")
            #print(data_df)
            cluster_value = predict_pipeline.add_clusters(self.age, self.work_experience, self.household_size, self.living_standards)
            data_df["Cluster"] = float(cluster_value)
            #print("After Cluster")
            #print(data_df)
            data_df['Age_Group'] = pd.cut(data_df['Age'], bins=[15, 30, 45, 60, 75], labels=['15-30', '31-45', '46-60', '61-75'])
            data_df['Age_Group'] = data_df['Age_Group'].astype(object)
            #print("After Age Group")
            #print(data_df)
            return data_df
        except Exception as e:
            raise CustomException(e, sys)
