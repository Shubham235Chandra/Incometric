import streamlit as st
import os
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

logging.basicConfig(level=logging.INFO)

# Adding custom CSS to enhance the appearance of the app
st.markdown(
    """
    <style>
    /* General app styling */
    body {
        background-color: #f7f9fc;
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .main {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }

    /* Title styling */
    .css-1d391kg {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }

    /* Sidebar styling */
    .css-12oz5g7 {
        margin-top: 20px;
        text-align: center;
        font-size: 1.5em;
        color: #ffffff;
        background-color: #2c3e50;
        padding: 10px;
        border-radius: 10px;
    }

    /* Input fields styling */
    .st-af, .st-ag {
        background-color: #ffffff;
        color: #333;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }

    /* Button styling */
    .stButton button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 1.2em;
        border: none;
        cursor: pointer;
        margin-top: 10px;
    }

    .stButton button:hover {
        background-color: #0056b3;
    }

    /* Success message styling */
    .stAlert {
        background-color: #28a745;
        color: white;
        border-radius: 5px;
        padding: 20px;
        font-size: 1.2em;
    }

    /* Error message styling */
    .stError {
        background-color: #dc3545;
        color: white;
        border-radius: 5px;
        padding: 20px;
        font-size: 1.2em;
    }

    /* Markdown text styling */
    .css-17xtpcr {
        font-size: 1.2em;
        line-height: 1.6;
        color: #333;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title('Incometric')

def validate_age(age):
    if age < 18:
        st.error('Age must be 18 or older.')
        return False
    return True

def main():
    show_basic_info = True

    with st.sidebar.form(key='user_info_form'):
        st.header('Enter your Information')

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox('Gender', ['Select your Gender', 'Male', 'Female'])
            primary_mode_of_transportation = st.selectbox('Primary Mode of Transportation', ['Select your Mode of Transportation', 'Public transit', 'Biking', 'Car', 'Walking'])
            education_level = st.selectbox('Education Level', ["Select your Education Level", "Master's", "Bachelor's", "High School", "Doctorate"])
            occupation = st.selectbox('Occupation', ['Select your Occupation', 'Technology', 'Finance', 'Education', 'Healthcare', 'Others'])
            marital_status = st.selectbox('Marital Status', ['Select your Marital Status', 'Married', 'Single', 'Divorced'])
            living_standards = st.selectbox('Living Standards', ['Select your Living Standards', 'High', 'Medium', 'Low'])
            homeownership_status = st.selectbox('Homeownership Status', ['Select your Homeownership Status', 'Own', 'Rent'])

        current_income = st.number_input('Your Current Yearly Income', min_value=0.0)

        with col2:
            location = st.selectbox('Location', ['Select your Location', 'Urban', 'Suburban', 'Rural'])
            type_of_housing = st.selectbox('Type of Housing', ['Select your Type of Housing', 'Apartment', 'Single-family home', 'Townhouse'])
            employment_status = st.selectbox('Employment Status', ['Select your Employment Status', 'Full-time', 'Self-employed', 'Part-time'])
            work_experience = st.number_input('Work Experience (in years)', min_value=0.0)
            number_of_dependents = st.number_input('Number of Dependents', min_value=0)
            household_size = st.number_input('Household Size', min_value=1)
            age = st.number_input('Age', min_value=18)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        show_basic_info = False
        if not validate_age(age):
            return

        try:
            logging.info("Received data for prediction")
            data = CustomData(
                primary_mode_of_transportation=primary_mode_of_transportation,
                education_level=education_level,
                occupation=occupation,
                marital_status=marital_status,
                living_standards=living_standards,
                gender=gender,
                homeownership_status=homeownership_status,
                location=location,
                type_of_housing=type_of_housing,
                employment_status=employment_status,
                work_experience=work_experience,
                number_of_dependents=number_of_dependents,
                household_size=household_size,
                age=age
            )

            pred_df = data.get_data_as_data_frame()

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predicts(pred_df)

            # Calculate the rounded result
            rounded_result = round(results[0])

            # Define an interval range (e.g., ±10% of the result)
            interval_range = 0.05 * rounded_result

            # Calculate the lower and upper bounds of the interval
            lower_bound = round((rounded_result - interval_range) / 1000) * 100
            upper_bound = round((rounded_result + interval_range) / 1000) * 100

            # Display the result in the specified format
            st.success(f'Based on the provided data, the predicted income range is between {int(lower_bound):,} and {int(upper_bound):,}.')

        except Exception as e:
            st.error(f"An error occurred: {e}")

    if show_basic_info:
        st.markdown("""
        #### Basic Information about the Inputs:
        - **Gender:** Select your gender from the options available.
        - **Primary Mode of Transportation:** Choose the primary mode of transportation you use.
        - **Education Level:** Select the highest level of education you have completed.
        - **Occupation:** Choose the sector in which you work.
        - **Marital Status:** Select your current marital status.
        - **Living Standards:** Choose the living standard that best describes your situation.
        - **Homeownership Status:** Select whether you own or rent your home.
        - **Location:** Choose the type of area where you live.
        - **Type of Housing:** Select the type of housing you reside in.
        - **Employment Status:** Choose your current employment status.
        - **Work Experience (in years):** Enter the number of years you have been working.
        - **Number of Dependents:** Enter the number of dependents you have.
        - **Household Size:** Enter the total number of people living in your household.
        - **Age:** Enter your age.
        """)

if __name__ == "__main__":
    main()
