import os
import sys
from dataclasses import dataclass


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.logger import logging as lg
from src.exception import CustomException
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            lg.info('Train-Test split initiated')

            X_train,y_train, X_test, y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])


            models = {
                "linear_regression": LinearRegression(),
                "random_forest_regressor": RandomForestRegressor(),
                "xgboost_regressor": XGBRegressor(),
                "catboost_regressor": CatBoostRegressor(verbose=False),
                "knn_regressor": KNeighborsRegressor(),
                "decision_tree_regressor": DecisionTreeRegressor(),
                "adaboost_regressor": AdaBoostRegressor(),
                "gradient_boosting_regressor": GradientBoostingRegressor()
            }

            lg.info('Initiating model trainer and model evaluation')
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            lg.info('Model training and evaluation completed')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.75:
                raise Exception('No best model found')
            lg.info('Best model found')

            lg.info('Saving best model')
            save_object(self.model_trainer_config.trained_model_path,best_model)
            lg.info('Best model saved')
            
            prediction = best_model.predict(X_test)
            accuracy = f'R2 score: {r2_score(y_test,prediction)}'

            return accuracy            
        
        except Exception as e:
            raise CustomException(e, sys)