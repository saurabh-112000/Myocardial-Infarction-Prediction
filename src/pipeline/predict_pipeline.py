import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 SEX: str,
                 ZSN_A: str,
                 L_BLOOD: float,
                 AGE: int,
                 ROE: float,
                 K_BLOOD: float,
                 AST_BLOOD: float,
                 NA_BLOOD: float,
                 ALT_BLOOD: float,
                 S_AD_ORIT: float,
                 D_AD_ORIT: float,
                 TIME_B_S: str,
                 lat_im: str,
                 STENOK_AN: str,
                 DLIT_AG: str,
                 ant_im: str,
                 inf_im: str,
                 INF_ANAM: str,
                 IBS_POST: str,
                 GB: str,
                 NA_R_1_n: str,
                 NOT_NA_1_n: str,
                 zab_leg_01: str,
                 FK_STENOK: str,
                 R_AB_1_n: str,
                 NA_R_2_n: str,
                 LID_S_n: str,
                 endocr_01: str,
                 NA_KB: str,
                 TRENT_S_n: str):

                self.SEX = SEX
                self.ZSN_A = ZSN_A
                self.L_BLOOD = L_BLOOD
                self.AGE = AGE
                self.ROE = ROE
                self.K_BLOOD = K_BLOOD
                self.AST_BLOOD = AST_BLOOD
                self.NA_BLOOD = NA_BLOOD
                self.ALT_BLOOD = ALT_BLOOD
                self.S_AD_ORIT = S_AD_ORIT
                self.D_AD_ORIT = D_AD_ORIT
                self.TIME_B_S = TIME_B_S
                self.lat_im = lat_im
                self.STENOK_AN = STENOK_AN
                self.DLIT_AG = DLIT_AG
                self.ant_im = ant_im
                self.inf_im = inf_im
                self.INF_ANAM = INF_ANAM
                self.IBS_POST = IBS_POST
                self.GB = GB
                self.NA_R_1_n = NA_R_1_n
                self.NOT_NA_1_n = NOT_NA_1_n
                self.zab_leg_01 = zab_leg_01
                self.FK_STENOK = FK_STENOK
                self.R_AB_1_n = R_AB_1_n
                self.NA_R_2_n = NA_R_2_n
                self.LID_S_n = LID_S_n
                self.endocr_01 = endocr_01
                self.NA_KB = NA_KB
                self.TRENT_S_n = TRENT_S_n


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
            "SEX": [self.SEX],
            "ZSN_A": [self.ZSN_A],
            "L_BLOOD": [self.L_BLOOD],
            "AGE": [self.AGE],
            "ROE": [self.ROE],
            "K_BLOOD": [self.K_BLOOD],
            "AST_BLOOD": [self.AST_BLOOD],
            "NA_BLOOD": [self.NA_BLOOD],
            "ALT_BLOOD": [self.ALT_BLOOD],
            "S_AD_ORIT": [self.S_AD_ORIT],
            "D_AD_ORIT": [self.D_AD_ORIT],
            "TIME_B_S": [self.TIME_B_S],
            "lat_im": [self.lat_im],
            "STENOK_AN": [self.STENOK_AN],
            "DLIT_AG": [self.DLIT_AG],
            "ant_im": [self.ant_im],
            "inf_im": [self.inf_im],
            "INF_ANAM": [self.INF_ANAM],
            "IBS_POST": [self.IBS_POST],
            "GB": [self.GB],
            "NA_R_1_n": [self.NA_R_1_n],
            "NOT_NA_1_n": [self.NOT_NA_1_n],
            "zab_leg_01": [self.zab_leg_01],
            "FK_STENOK": [self.FK_STENOK],
            "R_AB_1_n": [self.R_AB_1_n],
            "NA_R_2_n": [self.NA_R_2_n],
            "LID_S_n": [self.LID_S_n],
            "endocr_01": [self.endocr_01],
            "NA_KB": [self.NA_KB],
            "TRENT_S_n": [self.TRENT_S_n],
        }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)

         


