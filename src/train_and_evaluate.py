# load the train and test
# train algo
# save the metrices, params
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, plot_roc_curve, recall_score, precision_score
from get_data import read_params
import matplotlib.pyplot as plt
import argparse
import joblib
import json


# def eval_metrics(actual, pred):
#     rmse = np.sqrt(mean_squared_error(actual, pred))
#     mae = mean_absolute_error(actual, pred)
#     r2 = r2_score(actual, pred)
#     return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    # alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    # l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    #train_x = train.drop('famhist', axis=1)

    test_x = test.drop(target, axis=1)
    #test_x = test.drop('famhist', axis=1)

    # Build logistic regression model
    model = LogisticRegression(solver='lbfgs', random_state=0).fit(train_x, train_y)

    # Report training set score
    train_score = model.score(train_x, train_y) * 100
    print(train_score)
    # Report test set score
    test_score = model.score(test_x, test_y) * 100
    print(test_score)

    # Print classification report
    print(classification_report(test_y, model.predict(test_x)))

    # Confusion Matrix and plot
    cm = confusion_matrix(test_y, model.predict(test_x))
    print(cm)

    # fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    # auc = metrics.auc(fpr, tpr))

    roc_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    print(roc_auc)

    # recall = recall_score(test_y, y_pred, average='weighted')
    # print(recall)
    # precision = precision_score(test_y, y_pred, average='weighted')
    # print(precision)


    # Plot the ROC curve
    # model_ROC = plot_roc_curve(model, test_x, test_y)
    # print(model_ROC)



    #---------------------------------------------------------------------------------------------------------------------

    # # Plot outputs
    # plt.scatter( predicted_qualities,test_y,  color='black')
    # #plt.plot(test_y, predicted_qualities, color='blue', linewidth=3)
    # plt.show()
    
    # (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    # print("  RMSE: %s" % rmse)
    # print("  MAE: %s" % mae)
    # print("  R2: %s" % r2)

############################################################
    scores_file = config["reports"]["scores"]
    
    with open(scores_file, "w") as f:
        scores = {
            "train_score": train_score,
            "test_score": test_score,
            "roc_auc": roc_auc
            #"precision": precision
            # "recall": recall
                       
            
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

