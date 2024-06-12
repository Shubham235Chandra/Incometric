from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import logging

logging.basicConfig(level=logging.INFO)

application = Flask(__name__)

app = application

## Route for a Home Page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            logging.info("Received data for prediction")
            data = CustomData(
                primary_mode_of_transportation=request.form.get('primary_mode_of_transportation'),
                education_level=request.form.get('education_level'),
                occupation=request.form.get('occupation'),
                marital_status=request.form.get('marital_status'),
                living_standards=request.form.get('living_standards'),
                gender=request.form.get('gender'),
                homeownership_status=request.form.get('homeownership_status'),
                location=request.form.get('location'),
                type_of_housing=request.form.get('type_of_housing'),
                employment_status=request.form.get('employment_status'),
                work_experience=float(request.form.get('work_experience')),
                number_of_dependents=float(request.form.get('number_of_dependents')),
                household_size=float(request.form.get('household_size')),
                age=float(request.form.get('age'))
            )

            pred_df = data.get_data_as_data_frame()

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predicts(pred_df)

            logging.info(f"Prediction results: {results}")

            return render_template('home.html', results=results[0])
        except Exception as e:
            logging.error(f"Error in predict_datapoint: {e}")
            return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
