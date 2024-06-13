import streamlit as st
import os 

import google.generativeai as genai

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

from src.genai import advice

logging.basicConfig(level=logging.INFO)

st.title('Incometric')

def validate_age(age):
    if age < 18:
        st.error('Age must be 18 or older.')
        return False
    return True


def advice(current_income, predicted_income, gender, primary_mode_of_transportation, education_level, 
                            occupation, marital_status, living_standards, homeownership_status, location, type_of_housing, 
                            employment_status, work_experience, number_of_dependents, household_size):
    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('models/gemini-pro')
    
    input_details = f'''
    Hey, simulate an Expert Personal Financial advisor equipped with expertise in economic forecasting and demographic analysis. 
    Your task is to provide personalized financial advice based on the user's current income relative to a predicted value, 
    considering a comprehensive set of personal details. Based on the income difference, categorize the user's financial health 
    and offer targeted advice for income improvement.

    Current Income: {current_income}
    Predicted Income: {predicted_income}
    User Details:
    "Gender": "{gender}",
    "Primary Mode of Transportation": "{primary_mode_of_transportation}",
    "Education Level": "{education_level}",
    "Occupation": "{occupation}",
    "Marital Status": "{marital_status}",
    "Living Standards": "{living_standards}",
    "Homeownership Status": "{homeownership_status}",
    "Location": "{location}",
    "Type of Housing": "{type_of_housing}",
    "Employment Status": "{employment_status}",
    "Work Experience": "{work_experience} years",
    "Number of Dependents": "{number_of_dependents}",
    "Household Size": "{household_size}"
    Response format: {{ "Financial Health Category": "", "Advice": [ "Immediate Actions", "Gradual Changes" ] }}
    '''

    # Assuming `model.generate_content()` is a method that takes a string and generates content based on it
    output = model.generate_content(input_details)
    return output.text
    

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

        st.write("###")  # Adding space between the columns and the new input

        current_income = st.number_input('Your Current Yearly Income', min_value=0.0)

        st.write("###")  # Adding space between the new input and the next column

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
            predicted_income = round(results[0])

            st.success(f'The prediction is: {round(results[0])}')
            
            prompt = advice(current_income, predicted_income, gender, primary_mode_of_transportation, education_level, 
                            occupation, marital_status, living_standards, homeownership_status, location, type_of_housing, 
                            employment_status, work_experience, number_of_dependents, household_size)
            
            st.write(prompt)

            
        except Exception as e:
            pass

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
