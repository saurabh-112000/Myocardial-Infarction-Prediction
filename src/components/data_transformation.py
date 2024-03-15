import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent)) 
import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.logger import logging
from sklearn.impute import KNNImputer
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        #feature importances were computed and top 30 features are used for training ( view EDA Notebook )
        #this step was not included because, feature importances is a research process, and is not automated everyday, 
        #it requires periodical reviews, especially in health use cases.
        try:
            numerical_columns = ['L_BLOOD','ROE','K_BLOOD','AST_BLOOD','NA_BLOOD','ALT_BLOOD','S_AD_ORIT',
                                 'D_AD_ORIT','AGE']

            ordinal_columns = ['ZSN_A','TIME_B_S','lat_im','STENOK_AN','DLIT_AG','ant_im','inf_im','INF_ANAM','IBS_POST'
                               ,'GB','NA_R_1_n','NOT_NA_1_n','zab_leg_01','FK_STENOK', 'R_AB_1_n','NA_R_2_n']
            
            nominal_columns = ['LID_S_n','endocr_01','NA_KB','TRENT_S_n','SEX']

            num_pipeline = Pipeline(steps=[("imputer",KNNImputer(n_neighbors=5))])
            cat_ordinal_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent"))])
            cat_nominal_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent"))])
            logging.info(f"Categorical columns ( Ordinal ): {ordinal_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns ( Nominal ): {nominal_columns}")
            
            preprocessor=ColumnTransformer([
                
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_ordinal_pipeline",cat_ordinal_pipeline,ordinal_columns),
                ("cat_nominal_pipeline",cat_nominal_pipeline,nominal_columns)])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        



    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)


            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            target_column_name = "ZSN" 
            input_feature_train_df = train_df[['L_BLOOD','ROE','K_BLOOD','AST_BLOOD','NA_BLOOD','ALT_BLOOD','S_AD_ORIT',
                                 'D_AD_ORIT','AGE','ZSN_A','TIME_B_S','lat_im','STENOK_AN','DLIT_AG','ant_im','inf_im','INF_ANAM','IBS_POST'
                                ,'GB','NA_R_1_n','NOT_NA_1_n','zab_leg_01','FK_STENOK', 
                               'R_AB_1_n','NA_R_2_n','LID_S_n','endocr_01','NA_KB','TRENT_S_n','SEX']]
            target_feature_train_df = train_df[target_column_name]
            print(input_feature_train_df.shape)

            input_feature_test_df = test_df[['L_BLOOD','ROE','K_BLOOD','AST_BLOOD','NA_BLOOD','ALT_BLOOD','S_AD_ORIT',
                                 'D_AD_ORIT','AGE','ZSN_A','TIME_B_S','lat_im','STENOK_AN','DLIT_AG','ant_im','inf_im','INF_ANAM','IBS_POST'
                                ,'GB','NA_R_1_n','NOT_NA_1_n','zab_leg_01','FK_STENOK', 
                               'R_AB_1_n','NA_R_2_n','LID_S_n','endocr_01','NA_KB','TRENT_S_n','SEX']]
            

            
            print(input_feature_test_df.shape)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")


            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            missing_values_in_train = np.isnan(input_feature_train_arr).any()
            missing_values_in_test = np.isnan(input_feature_test_arr).any()
            print(f"Missing values in training data: {missing_values_in_train}")
            print(f"Missing values in testing data: {missing_values_in_test}")
            logging.info(f"Saved preprocessing object.")
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        

#if __name__ == "__main__":
    #obj = DataIngestion()
    #train_data,test_data = obj.initiate_data_ingestion()
    #data_transformation = DataTransformation()
    #data_transformation.initiate_data_transformation(train_data,test_data)