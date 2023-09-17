import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.logger import logging as lg

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_dataset_path: str = os.path.join('artifacts','train.csv')
    test_dataset_path: str = os.path.join('artifacts','test.csv')
    raw_dataset_path: str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        try:
            self.ingestion_config = DataIngestionConfig()
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self):
        lg.info('Initiating data ingestion')
        try:
            data = pd.read_csv('notebook/data/StudentsPerformance.csv') #~ This is the method where we can change the data source
            lg.info('Data ingestion completed')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_dataset_path), exist_ok=True)

            data.to_csv(self.ingestion_config.raw_dataset_path, index=False, header=True)

            lg.info('Train test split initiated')
            train_set,test_set = train_test_split(data,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_dataset_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_dataset_path, index=False, header=True)

            lg.info('Train test split completed')

            return (
                self.ingestion_config.train_dataset_path,
                self.ingestion_config.test_dataset_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    train_data,test_data = DataIngestion().initiate_data_ingestion()
    
    train_ar,test_ar,_ = DataTransformation().initiate_data_transformation(train_data,test_data)

    print(ModelTrainer().initiate_model_trainer(train_ar,test_ar))