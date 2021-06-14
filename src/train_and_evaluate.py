
  
# load the train and test
# train algo
# save the metrices, params
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

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    
    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)
    

    # Build logistic regression model
    model = LogisticRegression(solver='saga', random_state=0).fit(train_x, train_y)
    
    # Report training set score
    train_score = model.score(train_x, train_y) * 100
    print(train_score)
    # Report test set score
    test_score = model.score(test_x, test_y) * 100
    print(test_score)

    predicted_val = model.predict(test_x)

    #-------------------------------------------------------------------------------------------------------------------------
       
    precision, recall, prc_thresholds = metrics.precision_recall_curve(test_y, predicted_val)
    # print('precision value:', precision)
    # print('recall value:', recall)
    fpr, tpr, roc_thresholds = metrics.roc_curve(test_y, predicted_val)

    avg_prec = metrics.average_precision_score(test_y, predicted_val)
    roc_auc = metrics.roc_auc_score(test_y, predicted_val)

    scores_file = config["reports"]["scores"]
    prc_file = config["reports"]["prc"]
    roc_file = config["reports"]["roc"]
    auc_file = config["reports"]["auc"]
    # cm_file = config["reports"]["cm"]

    # with open(scores_file, "w") as fd:
    #     json.dump({"avg_prec": avg_prec, "roc_auc": roc_auc}, fd, indent=4)

    
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
        

    #------------------------------------------------------------------------------------------------------------------------

    # Print classification report
    print(classification_report(test_y, predicted_val))

    # Confusion Matrix and plot
    cm = confusion_matrix(test_y, predicted_val)
    print(cm)

        
    df1 = pd.DataFrame(predicted_val, columns = ['Predicted'])
    df_cm = pd.concat([test_y, df1], axis=1)
    print(df_cm)
    
    # with open(cm_file, "w") as fd:
    #     df1 = pd.DataFrame(predicted_val, columns = ['Predicted'])
    #     df_cm = pd.concat([test_y, df1], axis=1)
    #     # print(df_cm)
        
    df_cm.to_csv('cm.csv', index = False)

    # with open(auc_file, "w") as fd:
    #     json.dump(df_cm.to_json(), fd, indent=4, cls=NumpyEncoder)

    roc_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    print('ROC_AUC:{0:0.2f}'.format(roc_auc))

    Logistic_Accuracy = accuracy_score(test_y, predicted_val)
    print('Logistic Regression Model Accuracy:{0:0.2f}'.format(Logistic_Accuracy))

    # Average precision score
    average_precision = average_precision_score(test_y, predicted_val)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))


    # metrics.plot_roc_curve(model, test_x, test_y) 
    # plt.show() 

    #---------------- Random Forest -------------------------------

    # model_rf = RandomForestClassifier(n_estimators = 50)  
    
    # model_rf.fit(train_x, train_y)
    
    # # performing predictions on the test dataset
    # pred_rf = model_rf.predict(test_x)
    
    # # metrics are used to find accuracy or error
        
    
    # # using metrics module for accuracy calculation
    # RF_Accuracy = metrics.accuracy_score(test_y, pred_rf)
    # print("Random Forest Accuracy: ", RF_Accuracy)

    #---------------------------------------------------------------------

    # scores_file = config["reports"]["scores"]
    
    with open(scores_file, "w") as f:
        scores = {
            "train_score": train_score,
            "test_score": test_score,
            "roc_auc": roc_auc,
            #"Precision": precision,
            #"Recall": recall,            "Average precision": average_precision,
            "Logistic Accuracy": Logistic_Accuracy
            # "Random Forest Accuracy": RF_Accuracy                                 
            
        }
        json.dump(scores, f, indent=4)


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(model, model_path)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path = parsed_args.config)