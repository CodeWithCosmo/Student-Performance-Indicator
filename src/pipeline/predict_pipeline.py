import sys
import pandas as pd
from src.logger import logging as lg
from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            lg.info('Initiating prediction')
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            scaled_data = preprocessor.transform(features)

            prediction = model.predict(scaled_data)
            lg.info('Prediction completed')

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score) -> None:
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def to_dataframe(self):
        try:
            lg.info('Creating dataframe')
            input_dict = {
                'gender': [self.gender],
                'race_ethnicity': [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch': [self.lunch],
                'test_preparation_course': [self.test_preparation_course],
                'reading_score': [self.reading_score],
                'writing_score': [self.writing_score]                
            }
            lg.info('Dataframe created')
            dataframe = pd.DataFrame(input_dict)
            return dataframe
        except Exception as e:
            raise CustomException(e, sys)
