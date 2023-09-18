import pickle
import sys
from flask import Flask, request,render_template

# import numpy as np
# import pandas as pd


from src.logger import logging as lg
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictionPipeline,CustomData

# from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():  
         try: 
            data = CustomData(
                gender = request.form.get('gender'),
                race_ethnicity = request.form.get('ethnicity'),
                parental_level_of_education = request.form.get('parental_level_of_education'),
                lunch = request.form.get('lunch'),
                test_preparation_course = request.form.get('test_preparation_course'),
                reading_score = request.form.get('reading_score'),
                writing_score = request.form.get('writing_score')              
            )
            pred_df = data.to_dataframe()
            #  print(pred_df)

            predict_pipeline = PredictionPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template("home.html",prediction_text=f"Your Predicted Maths Score is {round(results[0],2)}") 
         
         except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    lg.info('Application started')
    app.run()