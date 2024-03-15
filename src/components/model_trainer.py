import sys
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).parent.parent.parent)) 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "XGBoost":XGBClassifier()
            }

            parameters = {
            
                "Decision Tree":
                {'max_depth' : [3,5,7,9,10,15,20,25],
                 'criterion' : ['gini', 'entropy'],
                 'max_features' : ['sqrt', 'log2'],
                'min_samples_split' : [2,4,6]
             },
                "Random Forest":
                { "n_estimators":[int(x) for x in np.linspace(start = 10, stop = 80, num = 10)],
                "max_features":['sqrt'],
                "max_depth":[2,4,5],
                "min_samples_split":[2, 5,7],
                "min_samples_leaf":[1, 2,5],
                "bootstrap":[True, False]
             }, 
                "Gradient Boosting": 
                { 'learning_rate': [0.01, 0.1, 0.2],
                 'subsample': [0.5, 0.75],
                 'criterion': ['friedman_mse'],
                 'min_samples_split': [2, 4, 8],
                 'min_samples_leaf': [1, 2, 3],
                 'max_depth': [3]
             },
                "AdaBoost": {
                 'n_estimators': [50, 100, 200],
                 'learning_rate': [0.01, 0.1, 1.0],
                 'algorithm': ['SAMME', 'SAMME.R']
             },
                "XGBoost": 
                {
                 'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4],
                'min_child_weight': [1, 3],
                 'subsample': [0.5],
                'colsample_bytree': [0.5, 0.7]
                }
            
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=parameters)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            print(best_model_name)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy
        
        except Exception as e:
            raise CustomException(e,sys)