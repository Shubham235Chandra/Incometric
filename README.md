# Incometric: The Future of Income Prediction

## Project Overview

Incometric is a data-driven application designed to predict potential income based on various personal and demographic factors. This project utilizes machine learning models and clustering techniques to provide users with an estimated income range and personalized financial advice. The application is built using Flask, Streamlit, and integrates Google Generative AI for advanced recommendations.

## Features

- **Income Prediction**: Predicts user's potential income based on their profile.
- **Clustering**: Uses K-means clustering to categorize users into different groups.
- **Personalized Financial Advice**: Provides actionable recommendations to improve financial stability and increase income.
- **User-Friendly Interface**: Streamlit-based UI for easy input and output visualization.

## Project Live 

1. **Basic Model**: This version runs the data science model to suggest your income based on the provided data. The suggestions are hard coded.
   - [Incometric Basic Model](https://incometric.streamlit.app/){:target="_blank"}

2. **Advanced Model**: This is a full-fledged advanced model where data science is integrated with Generative AI. It not only predicts income but also provides personalized recommendations from a Personal Financial Advisor.
   - [Incometric Advanced Model](https://huggingface.co/spaces/Shubham235/Incometric){:target="_blank"}

## Installation

### Prerequisites

- `Python 3.7` or `higher`
- Install required packages:

```bash
pip install -r requirements.txt
```

## Required Python Packages

To effectively set up the environment for your "Incometric" project, here's a more detailed guide on the required Python packages and their roles:

#### 1. **`pandas`** 
- Essential for data manipulation and analysis. Provides data structures and operations to manipulate numerical tables and time series.

#### 2. **`numpy`** 
- Adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

#### 3. **`seaborn`** 
- An advanced visualization library based on `matplotlib`. It provides a high-level interface for drawing attractive and informative statistical graphics.

#### 4. **`matplotlib`** 
- A plotting library for creating static, interactive, and animated visualizations in Python. It serves as the foundational graphical library for `seaborn`.

#### 5. **`scikit-learn`** 
- Offers simple and efficient tools for predictive data analysis. It's integral for machine learning, providing methods for classification, regression, clustering, and model evaluation.

#### 6. **`Flask`** 
- A micro web framework for Python. It's utilized to create a web server for your application, allowing for the deployment of your model as a web service.

#### 7. **`dill`** 
- Extends Python's `pickle` module. It's particularly useful for serializing and deserializing Python objects, especially those that are not supported by `pickle`.

#### 8. **`streamlit`** 
- Facilitates the creation of apps for machine machine learning and data science. It enables rapid development of data applications with minimal coding.

#### 9. **`python-dotboost`** 
- Reads key-value pairs from a `.env` file and sets them as environment variables. This helps in managing application secrets and configurations without hard coding them.

#### 10. **`google.generativeai`** 
- A package likely related to utilizing Google's Generative AI capabilities, such as those from the TensorFlow ecosystem or specific Google APIs focused on generative models.


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

### Project Files Overview

The Incometric application is organized into several key files, each serving a specific function within the project architecture:

#### 1. **`app.py`** 
- **Purpose:** Main Streamlit application file. Manages the user interface and integrates the entire prediction pipeline for user interactions.

#### 2. **`data_ingestion.py`** 
- **Purpose:** Manages the ingestion of data sources. Responsible for loading, cleaning, and splitting the data into training and testing sets.

#### 3. **`data_transformation.py`** 
- **Purpose:** Contains all data preprocessing and transformation functions. Essential for feature engineering and preparing data for modeling.

#### 4. **`model_trainer.py`** 
- **Purpose:** Handles the training and evaluation of machine learning models. Includes functionality for parameter tuning and cross-validation.

#### 5. **`kmeans_clustering.py`** 
- **Purpose:** Manages loading and prediction operations for the K-means clustering model. Useful for segmenting data or reducing dimensionality.

#### 6. **`predict_pipeline.py`** 
- **Purpose:** Includes the definition of the prediction pipeline class. Manages the flow from data input through to model prediction output.

#### 7. **`logger.py`** 
- **Purpose:** Sets up the logging configurations. Captures and logs runtime events and errors, aiding in debugging and monitoring.

#### 8. **`exception.py`** 
- **Purpose:** Manages custom exception handling. Enhances error management and control flow within the application.

#### 9. **`utils.py`** 
- **Purpose:** Contains utility functions for the application. Functions include object serialization/deserialization and additional model evaluation metrics.

#### 10. **`requirements.txt`** 
- **Purpose:** Lists all the necessary dependencies and their correct versions for the project. Ensures consistent setup across different environments.

### Prediction Pipeline

The prediction pipeline uses pre-trained models and preprocessing steps to transform user inputs and predict income. The pipeline includes:

- **CustomData Class**: Formats user input data.
- **PredictPipeline Class**: Loads models, preprocesses data, and makes predictions.
- **Clustering**: Enhances predictions by incorporating cluster information based on user data.

## Feature Engineering

Feature engineering played a critical role in improving the accuracy of the income prediction model. New feature columns were created, which significantly enhanced the model's performance. The most important engineered features include:

- **Living Standards**: This feature categorizes income into three levels: Low, Medium, and High. This categorization helps the model understand the user's financial situation relative to others.

  - `Low`: Income below a certain threshold.
  - `Medium`: Income within a middle range.
  - `High`: Income above a certain threshold.

- **Age Group**: This feature groups individuals into different age ranges. Age groups provide insights into different stages of life, which can correlate with income levels.

  - `15-30`: Young adults, early career stage.
  - `31-45`: Mid-career professionals.
  - `46-60`: Late-career professionals.
  - `61-75`: Nearing or in retirement.

- **Cluster**: This feature is derived from K-means clustering, which groups individuals into clusters based on similar demographic and socio-economic characteristics. Clustering helps the model leverage patterns from similar groups to make more accurate predictions.

## Dataset Overview

The original dataset used in this project is sourced from Kaggle: [Regression Dataset for Household Income Analysis](https://www.kaggle.com/datasets/stealthtechnologies/regression-dataset-for-household-income-analysis){:target="_blank"}. The dataset provides various demographic, socio-economic, and lifestyle factors that can influence household income.

### Original Dataset (`data.csv`)

The original dataset, `data.csv`, comprises various socio-economic and demographic attributes of individuals. This dataset includes the following columns:

### Columns:
- **`Age`**: The age of the individual in years (integer).
- **`Education_Level`**: The highest level of education attained (categorical: e.g., 'Master's', 'High School', 'Bachelor's').
- **`Occupation`**: The individual's occupation (categorical: e.g., 'Technology', 'Finance', 'Others').
- **`Number_of_Dependents`**: The number of dependents that the individual has (integer).
- **`Location`**: The type of area the individual resides in (categorical: e.g., 'Urban').
- **`Work_Experience`**: The number of years of professional experience (integer).
- **`Marital_Status`**: The marital status of the individual (categorical: e.g., 'Married', 'Single').
- **`Employment_Status`**: The current employment status (categorical: e.g., 'Full-time', 'Self-employed').
- **`Household_Size`**: The total number of people living in the household (integer).
- **`Income`**: Annual income in USD (integer).
- **`Homeownership_Status`**: Homeownership status (categorical: e.g., 'Own').
- **`Type_of_Housing`**: Type of housing the individual resides in (categorical: e.g., 'Apartment', 'Single-family home', 'Townhouse').
- **`Gender`**: The gender of the individual (categorical: e.g., 'Male', 'Female').
- **`Primary_Mode_of_Transportation`**: The primary mode of transportation used by the individual (categorical: e.g., 'Public transit', 'Biking', 'Car', 'Walking').

The dataset is well-structured with a total of 10,000 entries, each representing an individual's profile. This comprehensive data collection is essential for understanding patterns and making predictions related to socio-economic trends.

## Processed Dataset (`data_modified.csv`)

The dataset `data_modified.csv` is the result of several data transformation steps applied to the original `data.csv`. These transformations enhance the dataset's utility for predictive modeling and analysis.

### Data Transformation Steps:

- **Data Cleaning**: Involved removing duplicate rows and handling any missing values to ensure data integrity.
- **Feature Engineering**: Introduced new features that provide additional insights and segmentation:
  - **Living_Standards**: Income categorized into three levels: Low, Medium, High, based on threshold values determined from the data distribution.
  - **Age_Group**: Ages grouped into ranges (15-30, 31-45, 46-60, 61-75) to facilitate demographic-based analysis.
  - **Cluster**: Application of K-means clustering to group individuals based on similar demographic and socio-economic characteristics.
- **Data Transformation**: Numerical features were scaled to normalize data ranges, and categorical features were encoded to convert them into a format suitable for machine learning models.

### Enhanced Dataset Features:

The processed dataset includes these new columns along with the original features:

- **`Age_Group`**: Demographic group based on age.
- **`Living_Standards`**: Categorical income levels indicating economic status.
- **`Cluster`**: Labels from K-means clustering representing similar characteristic groups.

### Columns in `data_modified.csv`

The `data_dev.csv` dataset contains a mix of original and newly engineered features that enrich the data for analysis and predictive modeling:

- **`Age`**: The age of the individual in years.
- **`Education_Level`**: The highest level of education attained by the individual. Examples include 'Master's', 'High School', 'Bachelor's'.
- **`Occupation`**: The occupation of the individual. Examples include sectors like 'Technology', 'Finance', 'Others'.
- **`Number_of_Dependents`**: The number of dependents relying on the individual's income.
- **`Location`**: The residential setting of the individual, categorized as 'Urban' or other types.
- **`Work_Experience`**: Total years of professional experience.
- **`Marital_Status`**: Marital status of the individual, e.g., 'Married', 'Single'.
- **`Employment_Status`**: Current employment condition, such as 'Full-time', 'Self-employed'.
- **`Household_Size`**: Number of people living in the same household.
- **`Income`**: Annual income of the individual in USD.
- **`Homeownership_Status`**: Home ownership status, such as 'Own' or 'Rent'.
- **`Type_of_Housing`**: Type of housing, including 'Apartment', 'Single-family home', 'Townhouse'.
- **`Gender`**: Gender of the individual, e.g., 'Male', 'Female'.
- **`Primary_Mode_of_Transportation`**: Main mode of transportation used by the individual, such as 'Public transit', 'Car', 'Walking'.
- **`Age_Group`** (New): Age ranges grouped into categories like 15-30, 31-45, 46-60, 61-75, to aid in demographic segmentation.
- **`Living_Standards`** (New): Economic status divided into 'Low', 'Medium', and 'High', based on income levels.
- **`Cluster`** (New): Resultant groups from K-means clustering, which classifies individuals based on similar socio-economic and demographic characteristics.

These enhancements make the dataset particularly valuable for in-depth analysis and machine learning applications aimed at predicting socio-economic behaviors. The processed dataset, data_modified.csv, contains these additional features along with the original columns, enhancing the model's ability to make accurate predictions. This structured approach not only enhances the data's descriptive power but also its predictive capabilities, making it highly useful for socio-economic trendanalysis and predictive modeling in the "Incometric" project.

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
- **Acquire the Data**: The data has been provided as a CSV file named `data.csv` and its processed version `data_modified.csv`.
- **Read the Documentation**: Infer context from the data itself as no additional documentation is available.
- **Define Objectives**: Clean the data, understand its structure and distribution, and uncover patterns or insights.

#### Step 2: Dataset Structure
- **Numerical Columns**:
  - `Age`: The age of the individual in years.
  - `Number_of_Dependents`: The number of dependents relying on the individual's income.
  - `Work_Experience`: Total years of professional experience.
  - `Household_Size`: Number of people living in the same household.
  - `Income`: Annual income of the individual in USD.
- **Categorical Columns**:
  - `Education_Level`: The highest level of education attained by the individual (e.g., 'Master's', 'High School', 'Bachelor's').
  - `Occupation`: The occupation of the individual (e.g., 'Technology', 'Finance', 'Others').
  - `Location`: The residential setting of the individual (e.g., 'Urban').
  - `Marital_Status`: Marital status of the individual (e.g., 'Married', 'Single').
  - `Employment_Status`: Current employment condition (e.g., 'Full-time', 'Self-employed').
  - `Homeownership_Status`: Home ownership status (e.g., 'Own', 'Rent').
  - `Type_of_Housing`: Type of housing (e.g., 'Apartment', 'Single-family home', 'Townhouse').
  - `Gender`: Gender of the individual (e.g., 'Male', 'Female').
  - `Primary_Mode_of_Transportation`: Main mode of transportation used by the individual (e.g., 'Public transit', 'Car', 'Walking').

### Step 3: Data Cleaning
- **Remove Duplicates**: Ensure there are no duplicate rows that might skew the analysis.
- **Handle Missing Values**: Address any missing values appropriately, either by imputation or removal.

### Step 4: Feature Engineering
- **Create New Features**: Enhance the dataset by creating new features such as:
  - `Living_Standards`: Categorized income into three levels: Low, Medium, High.
  - `Age_Group`: Grouped ages into ranges: 15-30, 31-45, 46-60, 61-75.
  - `Cluster`: Used K-means clustering to group individuals based on similar characteristics.

### Step 5: Data Transformation
- **Scale Numerical Features**: Normalize numerical data to bring all features onto a comparable scale.
- **Encode Categorical Features**: Convert categorical data into a numerical format suitable for machine learning models.

### Step 6: Uncover Patterns and Insights
- **Visualizations**: Use graphs and plots to identify trends, distributions, and relationships within the data.
  - **Histograms**: Understand the distribution of numerical features.
  - **Box Plots**: Identify outliers and visualize the spread of the data.
  - **Scatter Plots**: Examine relationships between pairs of numerical variables.
  - **Bar Charts**: Analyze the frequency distribution of categorical variables.

### Conclusion
This structured approach to EDA ensures a comprehensive understanding of the dataset, laying the foundation for building robust predictive models in the "Incometric" project. The combination of data cleaning, feature engineering, and visualization provides critical insights necessary for informed decision-making.

For a detailed walkthrough of the EDA process, please refer to the [EDA Notebook](https://github.com/Shubham235Chandra/Incometric/blob/main/EDA/data/EDA.ipynb){:target="_blank"}.

### Project Outcome
After running the application, users can input their personal and demographic details such as age, gender, education level, occupation, and more. The application will then predict their potential income range and provide personalized financial advice.

## Contributing
Feel free to submit issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
