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
                "decision_tree_regressor": DecisionTreeRegressor(),
                "random_forest_regressor": RandomForestRegressor(),
                "gradient_boosting_regressor": GradientBoostingRegressor(),
                "linear_regressor": LinearRegression(),
                "xgboost_regressor": XGBRegressor(),
                "catboost_regressor": CatBoostRegressor(verbose=False),
                "adaboost_regressor": AdaBoostRegressor(),
                "knn_regressor": KNeighborsRegressor()
            }

            params={
                "decision_tree_regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "random_forest_regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "gradient_boosting_regressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "liner_regressor":{
                    "fit_intercept": [True, False],
                    "positive": [True, False],
                    "copy_X": [True, False],
                    "n_jobs": [-1, 1, 2, 4]
                },
                "xgboost_regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "catboost_regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "adaboost_regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "knn_regressor" : {
                    "n_neighbors": [1, 3, 5, 7, 9]
                # "weights": ["uniform", "distance", "inverse distance"],
                # "metric": ["minkowski", "manhattan", "chebyshev"],
                # "algorithm": ["auto", "kd_tree", "ball_tree"]
                }
            }       

            lg.info('Hyperparameter tuning initiated')
            lg.info('Initiating model trainer and model evaluation')
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            lg.info('Hyperparameter tuning completed')
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