from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

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
            work_experience=int(request.form.get('work_experience')),
            number_of_dependents=int(request.form.get('number_of_dependents')),
            household_size=int(request.form.get('household_size')),
            age=int(request.form.get('age'))
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
