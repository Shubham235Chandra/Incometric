# Incometric: The Future of Income Prediction

## Project Overview

Incometric is a data-driven application designed to predict potential income based on various personal and demographic factors. This project utilizes machine learning models and clustering techniques to provide users with an estimated income range and personalized financial advice. The application is built using Flask, Streamlit, and integrates Google Generative AI for advanced recommendations.

## Features

- **Income Prediction**: Predicts user's potential income based on their profile.
- **Clustering**: Uses K-means clustering to categorize users into different groups.
- **Personalized Financial Advice**: Provides actionable recommendations to improve financial stability and increase income.
- **User-Friendly Interface**: Streamlit-based UI for easy input and output visualization.

## Live Demos

1. **Basic Model**: This version runs the data science model to suggest your income based on the provided data. The suggestions are hard coded.
   - [Incometric Basic Model](https://incometric.streamlit.app/)

2. **Advanced Model**: This is a full-fledged advanced model where data science is integrated with Generative AI. It not only predicts income but also provides personalized recommendations from a Personal Financial Advisor.
   - [Incometric Advanced Model](https://huggingface.co/spaces/Shubham235/Incometric)

## Installation

### Prerequisites

- Python 3.7 or higher
- Install required packages:

```bash
pip install -r requirements.txt
```

## Required Packages
To run the Incometric application, ensure the following Python packages are installed:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- Flask
- dill
- streamlit
- python-dotenv
- google.generativeai

## Usage

### Running the Application
1. Ensure all required dependencies are installed.
2. Set up the `GOOGLE_API_KEY` in your environment variables.
3. Run the Streamlit application using the following command:

```bash
streamlit run app.py
```

# Incometric

## Project Structure

### Files
- `app.py`: Main Streamlit application file for the user interface and integrating the prediction pipeline.
- `data_ingestion.py`: Manages data ingestion, including splitting into training and testing sets.
- `data_transformation.py`: Contains data preprocessing and transformation functions.
- `model_trainer.py`: Handles training and evaluating machine learning models.
- `kmeans_clustering.py`: Manages loading and prediction using the K-means clustering model.
- `predict_pipeline.py`: Includes the prediction pipeline class and data handling.
- `logger.py`: Sets up logging configurations for the project.
- `exception.py`: Manages custom exception handling for the project.
- `utils.py`: Contains utility functions for object saving/loading and model evaluation.
- `requirements.txt`: Lists all necessary project dependencies.

### Prediction Pipeline

The prediction pipeline uses pre-trained models and preprocessing steps to transform user inputs and predict income. The pipeline includes:

- **CustomData Class**: Formats user input data.
- **PredictPipeline Class**: Loads models, preprocesses data, and makes predictions.
- **Clustering**: Enhances predictions by incorporating cluster information based on user data.

## Feature Engineering

Feature engineering played a critical role in improving the accuracy of the income prediction model. New feature columns were created, which significantly enhanced the model's performance. The most important engineered features include:

- **Living Standards**: This feature categorizes income into three levels: Low, Medium, and High. This categorization helps the model understand the user's financial situation relative to others.

  - Low: Income below a certain threshold.
  - Medium: Income within a middle range.
  - High: Income above a certain threshold.

- **Age Group**: This feature groups individuals into different age ranges. Age groups provide insights into different stages of life, which can correlate with income levels.

  - 15-30: Young adults, early career stage.
  - 31-45: Mid-career professionals.
  - 46-60: Late-career professionals.
  - 61-75: Nearing or in retirement.

- **Cluster**: This feature is derived from K-means clustering, which groups individuals into clusters based on similar demographic and socio-economic characteristics. Clustering helps the model leverage patterns from similar groups to make more accurate predictions.

## Dataset Overview

The original dataset used in this project is sourced from Kaggle: Regression Dataset for Household Income Analysis. The dataset provides various demographic, socio-economic, and lifestyle factors that can influence household income.

### Original Dataset (`data.csv`)

The original dataset, `data.csv`, contains the following columns:

Columns:
- Age
- Number_of_Dependents
- Work_Experience
- Household_Size
- Income
- Education_Level
- Occupation
- Location
- Marital_Status
- Employment_Status
- Homeownership_Status
- Type_of_Housing
- Gender
- Primary_Mode_of_Transportation

### Processed Dataset (`data_modified.csv`)

The original dataset was processed and transformed to create `data_modified.csv`, which was used in the project. The steps involved in transforming the dataset include:

- **Data Cleaning**: Removing any duplicate rows and handling missing values.
- **Feature Engineering**: Creating new features such as Living_Standards, Age_Group, and Cluster.
  - Living_Standards: Categorized income into three levels: Low, Medium, High.
  - Age_Group: Grouped ages into ranges: 15-30, 31-45, 46-60, 61-75.
  - Cluster: Used K-means clustering to group individuals based on similar characteristics.
- **Data Transformation**: Scaling numerical features and encoding categorical features to prepare the data for machine learning models.

The processed dataset, `data_modified.csv`, contains these additional features along with the original columns, enhancing the model's ability to make accurate predictions.


### File Descriptions

#### `app.py`
This file initializes the Streamlit app, sets up the UI, and processes user inputs. It integrates the prediction pipeline and displays the results along with personalized financial advice.

#### `data_ingestion.py`
Handles the ingestion of raw data, including reading the dataset, splitting it into training and testing sets, and saving these datasets to files. Points:
- Ingests data from CSV file.
- Splits data into training and testing sets.
- Saves split data to specified file paths.

#### `data_transformation.py`
Defines the preprocessing steps for the data, including scaling numerical features and encoding categorical features. It also saves the preprocessing object for later use in the prediction pipeline. Points:
- Identifies numerical and categorical features.
- Applies scaling and encoding to respective features.
- Saves the preprocessor object.

#### `model_trainer.py`
Responsible for training multiple machine learning models, evaluating their performance, and selecting the best-performing model. The selected model is saved for use in the prediction pipeline. Points:
- Initializes various regression models.
- Performs hyperparameter tuning using GridSearchCV.
- Evaluates models and selects the best based on R^2 score.
- Saves the best model.

#### `kmeans_clustering.py`
Contains functions to load the K-means model and scaler, and to predict the cluster for a given set of user features. Points:
- Loads pre-trained K-means model and scaler.
- Maps input data to clusters based on user features.

#### `predict_pipeline.py`
Defines the PredictPipeline class which loads the model and preprocessor, transforms features, and makes predictions. It also includes the CustomData class to format user inputs into a DataFrame. Points:
- Loads pre-trained models and preprocessors.
- Transforms user input data.
- Predicts income based on transformed data.

#### `logger.py`
Configures logging settings for the project to record important events and errors. Points:
- Generates log file with timestamp.
- Ensures logs directory exists.
- Configures logging format and level.

#### `exception.py`
Custom exception handling for the project. Points:
- Defines custom exception class.
- Provides detailed error messages including file and line number.

#### `utils.py`
Utility functions for saving/loading objects and evaluating models. Points:
- Saves objects to file using pickle.
- Loads objects from file using pickle.
- Evaluates models using GridSearchCV and returns performance metrics.

#### `requirements.txt`
Lists all the Python packages required to run the project.

## Exploratory Data Analysis (EDA)

### Overview
Incometric aims to predict individuals' income based on various demographic, socio-economic, and lifestyle factors. This section outlines the steps taken during the Exploratory Data Analysis (EDA) to understand the data and uncover patterns and insights.

### Steps

#### Step 1: Understand the Dataset
- Acquire the Data: The data has been provided as a CSV file.
- Read the Documentation: Infer context from the data itself as no additional documentation is available.
- Define Objectives: Clean the data, understand its structure and distribution, uncover patterns or insights.

#### Step 2: Dataset Structure
- Numerical Columns:
  - Age
  - Number_of_Dependents
  - Work_Experience
  - Household_Size
  - Income
- Categorical Columns:
  - Education_Level
  - Occupation
  - Location
  - Marital_Status
  - Employment_Status
  - Homeownership_Status
  - Type_of_Housing
  - Gender
  - Primary_Mode_of_Transportation

#### Step 3: Data Cleaning
- The dataset appears free from missing values and duplicates based on initial examination.
- Perform thorough checks to confirm.
- Dataset contains no duplicate rows and no missing values, allowing direct progression to data profiling and exploratory visualization.

#### Step 4: Data Profiling
- Objective: Summarize the data and confirm data types.
- Tasks:
  - Confirm data types of each column.
  - Analyze the distribution of numerical and categorical variables.

#### Step 5: Exploratory Visualization
- Visualize distributions and relationships between variables.
- Use plots like histograms, box plots, scatter plots, and correlation matrices to understand the data better.

### Example
After running the application, users can input their personal and demographic details such as age, gender, education level, occupation, and more. The application will then predict their potential income range and provide personalized financial advice.

## Contributing
Feel free to submit issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
