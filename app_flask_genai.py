from flask import Flask, render_template, request, redirect, url_for, flash
import google.generativeai as genai
import os
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Replace 'your_google_api_key_here' with your actual Google API key
GOOGLE_API_KEY = "AIzaSyArGKUgx81exVJVlRTMQUO_DWffOK7nyT0"

logging.basicConfig(level=logging.INFO)

## Route for a Home Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home_genai.html')
    else:
        try:
            gender = request.form.get('gender')
            primary_mode_of_transportation = request.form.get('primary_mode_of_transportation')
            education_level = request.form.get('education_level')
            occupation = request.form.get('occupation')
            marital_status = request.form.get('marital_status')
            living_standards = request.form.get('living_standards')
            homeownership_status = request.form.get('homeownership_status')
            location = request.form.get('location')
            type_of_housing = request.form.get('type_of_housing')
            employment_status = request.form.get('employment_status')
            work_experience = float(request.form.get('work_experience'))
            number_of_dependents = int(request.form.get('number_of_dependents'))
            household_size = int(request.form.get('household_size'))
            age = int(request.form.get('age'))
            current_income = float(request.form.get('current_income'))

            if age < 18:
                flash('Age must be 18 or older.')
                return redirect(url_for('home'))

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

            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel('models/gemini-pro')
            output = model.generate_content(input_details)

            return render_template('home_genai.html', results=predicted_income, suggestions=output[0]['generated_text'])

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            flash("An error occurred during prediction. Please try again.")
            return redirect(url_for('home_genai'))

    return render_template('home_genai.html')

if __name__ == "__main__":
    app.run(debug=True)
