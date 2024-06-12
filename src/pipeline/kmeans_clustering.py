import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import os
import sys


def load_model_and_scaler():
    # Assuming you have saved your model and scaler using joblib
    model_path = os.path.join('artifacts', 'kmeans_model.pkl')
    scaler_path = os.path.join('artifacts', 'scaler.pkl')
    model = load_object(file_path=model_path)
    scaler = load_object(file_path=scaler_path)
    
    return model, scaler

def predict_cluster(age, work_experience, household_size, living_standards):

    try:
        # Load the model and scaler
        model, scaler = load_model_and_scaler()
        
        # Create the DataFrame using a dictionary
        input_data = pd.DataFrame({
            'Age': [age], 
            'Work_Experience': [work_experience], 
            'Household_Size': [household_size], 
            'Income': [living_standards]
        })

        # Define the mapping dictionary
        income_mapping = {"Low": 64207.0, "Medium": 77808.0, "High": 2485100.0}

        # Apply the mapping
        input_data['Income'] = input_data['Income'].map(income_mapping)

        # Standardize the features
        X_scaled = scaler.transform(input_data)
        
        # Predict the cluster
        predict = model.predict(X_scaled)
        
        return predict[0]
    
    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)




'''
def clustering():
    # Load and map data
    data = pd.read_csv('kmeans.csv')

    # Standard Scaling 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, scaler, kmeans, clusters

def predict_cluster(age, work_experience, household_size, living_standards):

    # Set up the model and scaler
    model, scaler, kmeans, clusters = clustering()
    
    # Create the DataFrame using a dictionary
    input_data = pd.DataFrame({'age': age, 'work_experience': work_experience, 'household_size': household_size, 'living_standards': living_standards})

    # Define the mapping dictionary
    income_mapping = {"Low": 64207.0, "Medium": 77808.0, "High": 2485100.0}

    # Apply the mapping
    input_data['living_standards'] = input_data['living_standards'].map(income_mapping)

    X_scaled = scaler.fit_transform(input_data)
    predict = kmeans.predict(X_scaled)
    return predict[0]

'''