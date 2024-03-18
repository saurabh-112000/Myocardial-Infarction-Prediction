 # Myocardial Infarction Complications Prediction using Machine Learning

**Myocardial Infarction (MI)**, commonly known as a heart attack, is a medical condition that occurs when blood flow to a part of the heart is blocked for a long enough time that part of the heart muscle is damaged or dies. 

MI presents a significant challenge in contemporary healthcare, characterized by its high first-year mortality rate. Globally, the prevalence of MI remains substantial, with its impact particularly pronounced among the urban populations of advanced nations due to stress, and dietary inconsistencies. 

The progression of MI varies among patients, manifesting either as uncomplicated or with complications that do not alter long-term outcomes. 

Predicting these complications poses a challenge, even for seasoned professionals, making the proactive identification and prevention of such complications a crucial aspect of MI management. 

In this project, I have trained various ensemble classifiers on the data to predict Chronic Heart Failure based on multiple clinical parameters. The project consists of an end-to-end pipeline structure, starting from exploratory data analysis, data ingestion, data transformation, model training, and finally deployment. 

## Authors

- [Github - @saurabh-112000](https://github.com/saurabh-112000)
- [LinkedIn - Saurabh Sonawane](https://www.linkedin.com/in/saurabh112000/)


## Application

➼ [Home Page](https://mi-complicationpred-saurabh-d6d0de207765.herokuapp.com/)

➼ [Prediction Page](https://mi-complicationpred-saurabh-d6d0de207765.herokuapp.com/predictdata)

## Data

Dataset from UC Irvine Machine Learning Repository is used - [Link](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications)

➼ **Dataset characteristics** - Multivariate

➼ **Subject Area** - Health and Medicine

➼ **Primary Task** - Classification

➼ **Feature types** - Real

➼ **Features** # - 111, **Instances** # - 1700


## Deployment Workflow with Docker, GitHub Actions, and Heroku

➼ The setup in this project harnesses Docker, GitHub Actions, and Heroku to automate the deployment of the Python application, ensuring a seamless and consistent delivery process.

➼ **Docker Containerization**: The application is containerized using Docker, defined by a Dockerfile. This ensures that the application and its environment are packaged together, facilitating consistency across different stages of development, testing, and production.

➼ **GitHub Repo**: The codebase, including the Dockerfile and requirements.txt, is hosted on this repo. It also contains a .github/workflows directory with a YAML file specifying the CI/CD workflow with GitHub Actions.

➼ **GitHub Actions**: GitHub Actions automates the deployment pipeline. When changes are pushed to a designated branch or merged via a pull request, GitHub Actions triggers a workflow that builds the Docker container, tests it, and prepares it for deployment.

➼ **Heroku Deployment**: The workflow includes steps to log in to Heroku's container registry, push the Docker container there, and then release it to my Heroku application. This process updates my application on Heroku with the latest version of the containerized application.

➼ **Automated Deployments**: With every update to the main branch, the entire process from Docker container building to deployment on Heroku is automated. This ensures that the latest changes are always live, with minimal manual intervention.

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



