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
- catboost
- xgboost
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

- **CustomData Class**: Formats user input data.
- **PredictPipeline Class**: Loads models, preprocesses data, and makes predictions.
- **Clustering**: Enhances predictions by incorporating cluster information based on user data.

## Feature Engineering

Important engineered features improving model accuracy:

- **Living Standards**: Categorizes income into Low, Medium, and High, helping the model assess financial situations.
- **Age Group**: Groups individuals into age ranges, providing insights correlating with income levels.
- **Cluster**: Uses K-means clustering to group similar demographic and socio-economic characteristics.

## Dataset Overview

### Original Dataset (`data.csv`)

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

Transformations:
- Data Cleaning: Removes duplicates and handles missing values.
- Feature Engineering: Adds new features like Living Standards, Age Group, and Cluster.
- Data Transformation: Scales numerical features and encodes categorical features.

## Exploratory Data Analysis (EDA)

### Steps

1. **Understand the Dataset**
   - Acquire and infer context from the data.
   - Define objectives like data cleaning and understanding structure.

2. **Data Cleaning**
   - Check and confirm no missing values or duplicate rows.

3. **Data Profiling**
   - Confirm data types.
   - Analyze distributions of numerical and categorical variables.

4. **Exploratory Visualization**
   - Visualize distributions and relationships using histograms, box plots, scatter plots, and correlation matrices.

## Example Usage

Users input personal details like age, gender, and occupation. The application predicts potential income ranges and offers personalized financial advice.

## Contributing

Feel free to submit issues or pull requests. For major changes, open an issue first to discuss proposed changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
