import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from src.exception import CustomException
from src.logger import logging

def load_kaggle_dataset_to_dataframe(file_name):
    """
    Downloads a dataset from Kaggle and loads it into a Pandas DataFrame.
    
    Parameters:
    dataset_owner (str): The owner of the Kaggle dataset.
    dataset_name (str): The name of the Kaggle dataset.
    file_name (str): The name of the file to load from the dataset.
    
    Returns:
    pd.DataFrame: The dataset loaded into a Pandas DataFrame.
    """
    # Initialize and authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    dataset_owner = "heptapod"
    dataset_name = "titanic"
    file_name = "titanic.csv"  # Adjust the file name as needed
    
    df = load_kaggle_dataset_to_dataframe(dataset_owner, dataset_name, file_name)
    print(df.head())


    # Define the download path
    download_path = "src/data"
    os.makedirs(download_path, exist_ok=True)
    
    # Download the dataset
    api.dataset_download_files(f"{dataset_owner}/{dataset_name}", path=download_path, unzip=True)
    
    # Load the dataset into a DataFrame
    file_path = os.path.join(download_path, file_name)
    df = pd.read_csv(file_path)
    
    # Clean up the temporary files
    for root, dirs, files in os.walk(download_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(download_path)
    
    return df

    
