from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent)) 
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            SEX=int(request.form.get('SEX')),
            ZSN_A=int(request.form.get('ZSN_A')),
            L_BLOOD=float(request.form.get('L_BLOOD')),
            AGE=int(request.form.get('AGE')),
            ROE=float(request.form.get('ROE')),
            K_BLOOD=float(request.form.get('K_BLOOD')),
            AST_BLOOD=float(request.form.get('AST_BLOOD')),
            NA_BLOOD=float(request.form.get('NA_BLOOD')),
            ALT_BLOOD=float(request.form.get('ALT_BLOOD')),
            S_AD_ORIT=float(request.form.get('S_AD_ORIT')),
            D_AD_ORIT=float(request.form.get('D_AD_ORIT')),
            TIME_B_S=int(request.form.get('TIME_B_S')),
            lat_im=int(request.form.get('lat_im')),
            STENOK_AN=int(request.form.get('STENOK_AN')),
            DLIT_AG=int(request.form.get('DLIT_AG')),
            ant_im=int(request.form.get('ant_im')),
            inf_im=int(request.form.get('inf_im')),
            INF_ANAM=int(request.form.get('INF_ANAM')),
            IBS_POST=int(request.form.get('IBS_POST')),
            GB=int(request.form.get('GB')),
            NA_R_1_n=int(request.form.get('NA_R_1_n')),
            NOT_NA_1_n=int(request.form.get('NOT_NA_1_n')),
            zab_leg_01=int(request.form.get('zab_leg_01')),
            FK_STENOK=int(request.form.get('FK_STENOK')),
            R_AB_1_n=int(request.form.get('R_AB_1_n')),
            NA_R_2_n=int(request.form.get('NA_R_2_n')),
            LID_S_n=int(request.form.get('LID_S_n')),
            endocr_01=int(request.form.get('endocr_01')),
            NA_KB=int(request.form.get('NA_KB')),
            TRENT_S_n=int(request.form.get('TRENT_S_n')),
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        print(results)
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)      
