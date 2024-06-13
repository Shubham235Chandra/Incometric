import google.generativeai as genai
import os 

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('models/gemini-pro')

def advice(current_income, predicted_income, gender, primary_mode_of_transportation, education_level, 
                            occupation, marital_status, living_standards, homeownership_status, location, type_of_housing, 
                            employment_status, work_experience, number_of_dependents, household_size):
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
    



'''

input_prompt = """
            Hey, simulate a personal financial advisor equipped with expertise in economic forecasting and demographic analysis. 
            Your task is to provide personalized financial advice based on the user's current income relative to a predicted value, considering a comprehensive set of personal details. 
            Based on the income difference, categorize the user's financial health and offer targeted advice for income improvement.

            - Current Income: {current_income}
            - Predicted Income: {predicted_income}
            - User Details:
                - "Gender": "{gender}",
                - "Primary Mode of Transportation": "{primary_mode_of_transportation}",
                - "Education Level": "{education_level}",
                - "Occupation": "{occupation}",
                - "Marital Status": "{marital_status}",
                - "Living Standards": "{living_standards}",
                - "Homeownership Status": "{homeownership_status}",
                - "Location": "{location}",
                - "Type of Housing": "{type_of_housing}",
                - "Employment Status": "{employment_status}",
                - "Work Experience": "{work_experience} years",
                - "Number of Dependents": "{number_of_dependents}",
                - "Household Size": "{household_size}",
                - "Age": "{age}"

            Response format:
            {{
            "Financial Health Category": "",
            "Advice": [
                "Immediate Actions",
                "Gradual Changes"
            ]
            }}
            """

            formatted_prompt = input_prompt.format(
                current_income=current_income,
                predicted_income=predicted_income,
                gender=gender,
                primary_mode_of_transportation=primary_mode_of_transportation,
                education_level=education_level,
                occupation=occupation,
                marital_status=marital_status,
                living_standards=living_standards,
                homeownership_status=homeownership_status,
                location=location,
                type_of_housing=type_of_housing,
                employment_status=employment_status,
                work_experience=work_experience,
                number_of_dependents=number_of_dependents,
                household_size=household_size,
                age=age
            )
            '''