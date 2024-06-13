import streamlit as st
import os
import google.generativeai as genai
import json
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

logging.basicConfig(level=logging.INFO)

# Initialize the GenerativeAI client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('models/gemini-pro')


## Streamlit App
# Setting a custom theme
st.set_page_config(page_title='Incometric', page_icon=':moneybag:', layout='wide')
st.title("💲 Incometric")

# Custom CSS for sidebar
st.markdown("""
    <style>
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    .css-1d391kg h1 {
        color: #2c3e50;
    }
    .css-1d391kg textarea {
        border-color: #2c3e50;
    }
    .css-1d391kg .stTextInput {
        border-color: #2c3e50;
    }
    .css-1d391kg .stButton button {
        background-color: #2c3e50;
        color: white;
        border: none;
    }
    .css-1d391kg .stButton button:hover {
        background-color: #34495e;
    }
    .container {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .header {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }
    .subheader {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
    }
    .text {
        font-size: 16px;
        color: #2c3e50;
    }
    .css-1d391kg .sidebar {
        position: absolute;
        right: 0;
        width: 300px;
        top: 0;
        background-color: #f0f2f6;
    }
    .css-1d391kg .sidebar-content {
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

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

        with col2:
            living_standards = st.selectbox('Living Standards', ['Select your Living Standards', 'High', 'Medium', 'Low'])
            homeownership_status = st.selectbox('Homeownership Status', ['Select your Homeownership Status', 'Own', 'Rent'])
            location = st.selectbox('Location', ['Select your Location', 'Urban', 'Suburban', 'Rural'])
            type_of_housing = st.selectbox('Type of Housing', ['Select your Type of Housing', 'Apartment', 'Single-family home', 'Townhouse'])
            employment_status = st.selectbox('Employment Status', ['Select your Employment Status', 'Full-time', 'Self-employed', 'Part-time'])

        col3, col4 = st.columns(2)

        with col3:
            work_experience = st.slider('Work Experience (in years)', min_value=0.0, max_value=75.0, step=0.1)
            number_of_dependents = st.number_entered = st.slider('Number of Dependents', min_value=0, max_value=25)

        with col4:
            household_size = st.number_entered = st.slider('Household Size', min_value=1, max_value=100)
            age = st.number_entered = st.slider('Age', min_value=18, max_value=110)
        
        # Current Income input
        current_income = st.number_input('Your Current Income in `$`', min_value=0.0)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        show_basic_info = False
        if not validate_age(age):
            return

        try:
            logging.info("Received data for prediction")
            user_data = {
                'primary_mode_of_transportation': primary_mode_of_transportation,
                'education_level': education_level,
                'occupation': occupation,
                'marital_status': marital_status,
                'living_standards': living_standards,
                'gender': gender,
                'homeownership_status': homeownership_status,
                'location': location,
                'type_of_housing': type_of_housing,
                'employment_status': employment_status,
                'work_experience': work_experience,
                'number_of_dependents': number_of_dependents,
                'household_size': household_size,
                'age': age
            }
            data = CustomData(**user_data)

            pred_df = data.get_data_as_data_frame()

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predicts(pred_df)

            rounded_result = round(results[0])
            interval_range = 0.05 * rounded_result
            lower_bound = round((rounded_result - interval_range) / 1000) * 100
            upper_bound = round((rounded_result + interval_range) / 1000) * 100
            
            # Display the result
            st.success(f"Based on your profile, your Income range should be in the range:   💲{int(lower_bound):,} and 💲{int(upper_bound):,}.")


            input_details = f'''
                Assume the role of an Expert Personal Financial Advisor with a deep understanding of economic forecasting and demographic analysis. 
                Your primary responsibility is to provide precise, personalized financial advice by assessing the user's current income relative to a projected figure. 
                Ensure that this analysis extensively utilizes a wide array of the user's personal and financial details to achieve the utmost accuracy and relevance.

                Evaluate the user's financial health, classifying it into a specific category that reflects their current situation, and provide a brief rationale for this classification. 
                Based on the difference between their existing and predicted income levels, offer five specific, constructive, and innovative recommendations aimed at enhancing their financial stability and increasing their income for both sections. 
                These should be divided into two categories: Immediate Actions and Gradual Changes.

                Each set of recommendations should be diverse, covering various aspects of life, including investment strategies, spending habits, career development, personal savings, and lifestyle adjustments.

                Your advice should be supportive and optimistic, incorporating the latest market trends and economic conditions to ensure practical, forward-thinking guidance that empowers the user to make informed financial decisions.
                
                Current Income: {current_income}
                Predicted Income: {rounded_result}
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

            # Using GEMINI PRO from GOOGLE to get personalized Financial Report
            output = model.generate_content(input_details)

            advice_dict = json.loads(output.text)


            with st.container():

                st.markdown('<div class="header">Personalized Recommendations to Enhance Your Financial Future</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="text">
                    Based on your profile and current financial situation, we've crafted some tailored advice to help you improve your income and achieve greater financial stability. 
                    These recommendations are divided into two categories: Immediate Actions and Gradual Changes. 
                    Implementing these suggestions can help you take control of your finances and work towards a more secure and prosperous future.
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f'<div class="subheader">Financial Health Category: <strong>{advice_dict["Financial Health Category"]}</strong></div>', unsafe_allow_html=True)

                st.markdown('<div class="subheader">Immediate Actions:</div>', unsafe_allow_html=True)
                st.markdown('<ul>', unsafe_allow_html=True)
                for action in advice_dict["Advice"]["Immediate Actions"]:
                    st.markdown(f'<li class="text">{action}</li>', unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)

                st.markdown('<div class="subheader">Gradual Changes:</div>', unsafe_allow_html=True)
                st.markdown('<ul>', unsafe_allow_html=True)
                for change in advice_dict["Advice"]["Gradual Changes"]:
                    st.markdown(f'<li class="text">{change}</li>', unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)

                st.markdown('</br>', unsafe_allow_html=True)
                st.markdown("""
                    <div class="text" style="margin-top: 20px;">
                        <blockquote style="font-size: 18px; color: #2c3e50; border-left: 5px solid #2c3e50; padding-left: 15px;">
                            <b>We encourage you to start with the immediate actions to make quick improvements, while also planning for gradual changes that will benefit you in the long run. 
                            Remember, achieving financial stability is a journey, and taking consistent, informed steps will help you reach your goals.</b>
                        </blockquote>
                    </div>
                    """, unsafe_allow_html=True)


        except Exception as e:
            logging.error("Error in prediction", exc_info=True)
            st.error("We encountered an issue while generating your recommendations using GENAI. The system is currently under too much load. Please reload the page and re-enter your information. If the problem persists, try again later or contact support for assistance.")


    if show_basic_info:
        st.markdown("""
        Welcome to Incometric! This app helps you predict your potential income based on various personal and demographic factors. 
        By entering your information in the sidebar, you can receive an estimated income range and personalized financial advice.

        ### How It Works:
        1. **Input Your Information**: Provide details about your gender, education, occupation, marital status, living standards, and more.
        2. **Receive Income Prediction**: Based on the information you provide, the app predicts your potential income range.
        3. **Get Personalized Advice**: The app offers actionable recommendations to help you improve your financial stability and increase your income.
        ### Start by entering your information in the sidebar form!
                    
        #### Basic Information about the Inputs:

        - **Gender:** Select your gender from the options available (e.g., Male or Female). This helps to personalize the income prediction and advice based on gender-specific trends and data.
        - **Primary Mode of Transportation:** Choose the primary mode of transportation you use for your daily commute (e.g., Public transit, Biking, Car, or Walking). This provides insight into your commuting costs and lifestyle.
        - **Education Level:** Select the highest level of education you have completed (e.g., High School, Bachelor's, Master's, or Doctorate). Education level significantly impacts earning potential and career opportunities.
        - **Occupation:** Choose the sector in which you work (e.g., Technology, Finance, Education, Healthcare, or Others). Different occupations have varying income levels and growth prospects.
        - **Marital Status:** Select your current marital status (e.g., Married, Single, or Divorced). Marital status can affect household income, tax benefits, and financial planning.
        - **Living Standards:** Choose the living standard that best describes your situation (e.g., High, Medium, or Low). This helps to gauge your cost of living and disposable income.
        - **Homeownership Status:** Select whether you own or rent your home (e.g., Own or Rent). Homeownership status impacts your long-term financial stability and asset accumulation.
        - **Location:** Choose the type of area where you live (e.g., Urban, Suburban, or Rural). Location influences your cost of living, job opportunities, and lifestyle.
        - **Type of Housing:** Select the type of housing you reside in (e.g., Apartment, Single-family home, or Townhouse). This information helps to assess your housing expenses and living conditions.
        - **Employment Status:** Choose your current employment status (e.g., Full-time, Self-employed, or Part-time). Employment status directly affects your income and job security.
        - **Work Experience (in years):** Enter the number of years you have been working. More years of work experience generally correlate with higher income levels and job expertise.
        - **Number of Dependents:** Enter the number of dependents you have. Dependents can influence your financial responsibilities and household expenses.
        - **Household Size:** Enter the total number of people living in your household. Household size affects your living expenses and financial planning.
        - **Age:** Enter your age. Age is an important factor in income prediction and financial advice, as it correlates with career stage, earning potential, and retirement planning.
        """)

if __name__ == "__main__":
    main()
