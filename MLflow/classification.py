import os
import warnings
import sys
import pandas as pd
import numpy as np
import yaml
import mlflow
import logging
import mlflow.sklearn
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, plot_roc_curve,accuracy_score, recall_score, precision_score


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# folder to load config file
CONFIG_PATH = "./"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("Config.yaml")


if __name__ == "__main__":
    
    # load data
    df = pd.read_csv(os.path.join(config["data_directory"], config["data_name"]))
    df = df.dropna()
    df = pd.get_dummies(df, columns = ['famhist'], drop_first=True)
    df.drop(config['drop_columns'],axis=1, inplace=True)

    # Split data into train and test
    train, test = train_test_split(df, test_size=config['test_size'], random_state = config['random_state'])

    train_x = train.drop(config['target_name'], axis=1)
    test_x = test.drop(config['target_name'], axis=1)
    train_y = train[config['target_name']]
    test_y = test[config['target_name']]
    

    with mlflow.start_run():

        model = LogisticRegression(solver= config['solver'], random_state=config['random_state']).fit(train_x, train_y)
        train_score = model.score(train_x, train_y) * 100
        print(train_score)
        test_score = model.score(test_x, test_y) * 100
        print(test_score)

        predicted_val = model.predict(test_x)

        roc_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
        accuracy = accuracy_score(test_y, predicted_val)
        
        # Generating model metrics
        cm = confusion_matrix(test_y, predicted_val)
        precision = precision_score(test_y, predicted_val, labels=[1,2], average='micro')
        recall = recall_score(test_y, predicted_val, average='micro')

        # Logging and tracking model metrics
        mlflow.log_param("Train Score", train_score)
        mlflow.log_param("Test Score", test_score)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        
        if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(model, "model", registered_model_name="Logistic Regression")
        else:
            mlflow.sklearn.log_model(model, "model")