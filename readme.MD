
## Sections

➼ **EDA.ipynb** has the entire exploratory data analysis that was done on the dataset, including missing value imputation and feature selection. 

➼ **Data Ingestion Module**

This module, defined in data_ingestion.py, automates the process of loading, splitting, and preparing the dataset for further stages in the machine learning pipeline. It is designed to ensure that the raw data is readily available for training and testing models without manual intervention.

➼ **Data Transformation Module**

The data_transformation.py script automates the preprocessing of data to enhance model training effectiveness. This module is designed to clean and encode the dataset, ensuring it is in the optimal format for feeding into machine learning models.

➼ **Model Training Module**

The model_trainer.py script encapsulates the process of training machine learning models, evaluating their performance, and selecting the best-performing model based on predefined metrics. This module integrates seamlessly with the data preparation phases, utilizing preprocessed datasets for training and testing.

Models tested: Decision Tree, Random Forest, Gradient Boosting, AdaBoost, and XGBoost.

Hyperparameter Tuning Method: Grid Search

➼ **Prediction Module**

The predict_pipeline.py script stores user inputed data and passes it on to preprocessor object which then leverages the trained model to make a prediction and return back the result. 

➼ **Application and Front end**: 

app.py creates the flask framework for the application by leveraging index.html and home.html and then calls the prediction module. 



