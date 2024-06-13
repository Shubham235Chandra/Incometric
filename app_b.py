import streamlit as st
import os
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

logging.basicConfig(level=logging.INFO)

# Adding custom CSS to enhance the appearance of the app
# Setting a custom theme
st.set_page_config(page_title='Incometric', page_icon=':moneybag:', layout='wide')
st.markdown("""
<style>
    .big-font {
        font-size:20px;
        font-weight: bold;
    }
    .stTextInput>label, .stSelectbox>label, .stNumberInput>label {
        font-weight: bold;
    }
</ol>
</style>
""", unsafe_allow_html=True)

st.title('Incometric')

def validate_age(age):
    if age < 18:
        st.error('Age must be 18 or older.')
        return False
    return True

def generate_recommendations(user_data, predicted_income):
    recommendations = []

    # Career and Professional Development
    if user_data['education_level'] in ["High School", "Bachelor's"]:
        more_education = "a Bachelor's degree" if user_data['education_level'] == "High School" else "a Master's or professional degree"
        recommendations.append(
            f"Consider pursuing {more_education}. Fields like {user_data['occupation'].lower()} are rapidly evolving, and advanced education can open up senior roles and increase earning potential."
        )
        if user_data['occupation'] == 'Technology':
            recommendations.append(
                "Enhance your tech skills by learning latest technologies such as AI, machine learning, or blockchain through specialized courses on platforms like Coursera or Udacity."
            )
    
    # Financial Advice
    if predicted_income < 50000:
        recommendations.append(
            "Maximize your savings by adhering to a budget that allocates expenses into needs, wants, and savings, potentially using the 50/30/20 rule to manage cash flow more effectively."
        )
        recommendations.append(
            "Consider setting up an emergency fund, ideally six months' worth of living expenses, to protect against unforeseen financial disruptions."
        )

    # Lifestyle Adjustments
    if user_data['marital_status'] == 'Married' and user_data['number_of_dependents'] > 0:
        recommendations.append(
            "Invest in a life insurance policy that provides adequate coverage for your dependents. Review and adjust it as your financial situation or family structure changes."
        )
        recommendations.append(
            "Balance work and life effectively to enhance quality of life. Explore flexible working arrangements if available, which might include telecommuting or flexible work hours."
        )

    # Housing and Living Environment
    if user_data['homeownership_status'] == 'Rent' and user_data['living_standards'] != 'High':
        investment_advise = "in real estate that could appreciate over time" if user_data['location'] == 'Suburban' else "in safer, low-risk bonds or mutual funds"
        recommendations.append(
            f"As a renter, consider saving for a down payment on a property or investing {investment_advise}, especially if your current living standards are not high."
        )
    
    # Personal Development and Community Engagement
    if user_data['location'] == 'Rural':
        recommendations.append(
            "Consider developing or participating in community-based projects that can enhance local amenities and increase property values, such as farmer's markets or community gardens."
        )

    # Digital Literacy and Connectivity
    if user_data['occupation'] in ['Education', 'Healthcare']:
        recommendations.append(
            "Stay updated with the latest digital tools and platforms that can enhance productivity and service delivery in your field. Regular training sessions can be beneficial."
        )

    # Long-term Planning
    recommendations.append(
        "Regularly review and adjust your financial plans and investments based on life changes, economic conditions, and personal goals. Consider consulting with a financial advisor annually."
    )

    # Health and Well-being
    if user_data['age'] > 40:
        recommendations_checkup = "biannual" if user_data['number_of_dependents'] > 2 else "annual"
        recommendations.append(
            f"Prioritize your health with regular {recommendations_checkup} health checkups, focusing on preventative care to avoid long-term health issues and reduce potential medical costs."
        )

    return recommendations


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

        with col2:
            location = st.selectbox('Location', ['Select your Location', 'Urban', 'Suburban', 'Rural'])
            type_of_housing = st.selectbox('Type of Housing', ['Select your Type of Housing', 'Apartment', 'Single-family home', 'Townhouse'])
            employment_status = st.selectbox('Employment Status', ['Select your Employment Status', 'Full-time', 'Self-employed', 'Part-time'])
            work_experience = st.number_input('Work Experience (in years)', min_value=0.0)
            number_of_dependents = st.number_input('Number of Dependents', min_value=0)
            household_size = st.number_input('Household Size', min_value=1)
            age = st.number_entered = st.number_input('Age', min_value=18)

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

            st.success(f'Based on your Profile, your Income Range should be between {int(lower_bound):,} and {int(upper_bound):,}.')

            # Generate recommendations
            recommendations = generate_recommendations(user_data, rounded_result)
            if recommendations:
                st.write("### Recommendations to Improve Your Income:")
                for recommendation in recommendations:
                    st.markdown("- " + recommendation)

        except Exception as e:
            logging.error("Error in prediction", exc_info=True)
            st.error(f"An error occurred during the prediction process: {str(e)}")

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