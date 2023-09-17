import sys 
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from src.logger import logging as lg
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_features = ['writing_score','reading_score']
            categorical_features = [
                'gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            
            numerical_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            lg.info('Numerical pipeline created')

            categorical_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder()), 
                ('scaler',StandardScaler(with_mean=False))              

            ])
            lg.info('Categorical pipeline created')
            
            preprocessor = ColumnTransformer(
                [('numerical',numerical_pipeline,numerical_features),
                 ('categorical',categorical_pipeline,categorical_features)]
            )
            lg.info('Preprocessor created')
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self,train_path:str,test_path:str):
        try:
            lg.info('Initiating data transformation')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            lg.info('Train and test dataframes loaded successfully')
            lg.info('Obtaining preprocessor')

            preprocessor = self.get_data_transformer()

            target_column = ['math_score']
            numerical_column = ['writing_score','reading_score']

            input_feature_train_df = train_df.drop(target_column,axis=1)
            target_feature_train_df = train_df[target_column]            

            input_feature_test_df = test_df.drop(target_column,axis=1)
            target_feature_test_df = test_df[target_column]

            lg.info('Applying preprocessor on train and test dataframes')

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            lg.info('Preprocessor applied on train and test dataframes')

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            #* NOTE: `np.c_` is a special object in NumPy that is used to concatenate arrays along their second axis.            
            
            lg.info('Saving preprocessor object')
            save_object(self.data_transformation_config.preprocessor_obj_file_path,preprocessor)
            lg.info('Preprocessor object saved')
            lg.info('Data transformation completed')

            return train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            
        except Exception as e:
            raise CustomException(e, sys)