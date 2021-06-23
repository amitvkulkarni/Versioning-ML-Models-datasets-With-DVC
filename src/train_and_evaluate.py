
  
import os
import math
import warnings
import sys
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, plot_roc_curve,accuracy_score
from sklearn.metrics import average_precision_score
from numpyencoder import NumpyEncoder
from get_data import read_params
import matplotlib.pyplot as plt
import argparse
import joblib
import json
import dvclive

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    solver = config["base"]["solver"]
    
    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    train_size = train.shape[0]
    print("Train Size", train_size)
    dvclive.log("Train", train_size)
    test = pd.read_csv(test_data_path, sep=",")
    test_size = test.shape[0]
    print("Test Size", test_size)
    dvclive.log("Test", test_size)

    
    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)
    

    # Build logistic regression model
    model = LogisticRegression(solver=solver, random_state=random_state).fit(train_x, train_y)
    
    # Report training set score
    train_score = model.score(train_x, train_y) * 100
    dvclive.log("Train Score", train_score)
    print(train_score)
    # Report test set score
    test_score = model.score(test_x, test_y) * 100
    dvclive.log("Test Score",test_score)
    print(test_score)

    predicted_val = model.predict(test_x)

           
    precision, recall, prc_thresholds = metrics.precision_recall_curve(test_y, predicted_val)
    fpr, tpr, roc_thresholds = metrics.roc_curve(test_y, predicted_val)

    avg_prec = metrics.average_precision_score(test_y, predicted_val)
    roc_auc = metrics.roc_auc_score(test_y, predicted_val)
    
    scores_file = config["reports"]["scores"]
    prc_file = config["reports"]["prc"]
    roc_file = config["reports"]["roc"]
    auc_file = config["reports"]["auc"]
    

        
    nth_point = math.ceil(len(prc_thresholds)/1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]    
    
    
    with open(prc_file, "w") as fd:
        prcs = {
                "prc": [
                    {"precision": p, "recall": r, "threshold": t}
                    for p, r, t in prc_points
                ]
            }
        json.dump(prcs, fd, indent=4, cls=NumpyEncoder)
        

    with open(roc_file, "w") as fd:
        rocs = {
                "roc": [
                    {"fpr": fp, "tpr": tp, "threshold": t}
                    for fp, tp, t in zip(fpr, tpr, roc_thresholds)
                ]
            }
        json.dump(rocs, fd, indent=4, cls=NumpyEncoder)
        

    

    # Print classification report
    print(classification_report(test_y, predicted_val))

    # Confusion Matrix and plot
    cm = confusion_matrix(test_y, predicted_val)
    print(cm)

        
    df1 = pd.DataFrame(predicted_val, columns = ['Predicted'])
    df_cm = pd.concat([test_y, df1], axis=1)
    print(df_cm)
    
          
    df_cm.to_csv('cm.csv', index = False)



    roc_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    dvclive.log("roc_auc", roc_auc)
    print('ROC_AUC:{0:0.2f}'.format(roc_auc))

    Logistic_Accuracy = accuracy_score(test_y, predicted_val)
    dvclive.log("Accuracy",Logistic_Accuracy)
    print('Logistic Regression Model Accuracy:{0:0.2f}'.format(Logistic_Accuracy))

    # Average precision score
    average_precision = average_precision_score(test_y, predicted_val)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    
    with open(scores_file, "w") as f:
        scores = {
            "Train Score": train_score,
            "Test Score": test_score,
            "ROC_AUC": roc_auc,
            "Train Size": train_size,
            "Test Size": test_size,
            "Solver": solver,
            "Accuracy": Logistic_Accuracy
            
            
        }
        json.dump(scores, f, indent=4)


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    
    joblib.dump(model, model_path)
    print("600 KB dataset with solver as saga")


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path = parsed_args.config)