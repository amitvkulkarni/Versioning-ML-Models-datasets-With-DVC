## read params
## process
### return dataframe
import os
import yaml
import pandas as pd
import argparse
import dvclive

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    data_path = config["data_source"]["s3_source"]
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    df = pd.get_dummies(df, columns = ['famhist'], drop_first=True)
    df.drop("sbp",axis=1, inplace=True)
    print("Dataset Size: ", df.shape[0])
    print("******************check for version control**********************")
    print(df.count)
    print("******************check for version control*****************************************")
    dvclive.log("# Records",df.shape[0])
    return df

###

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path = parsed_args.config)