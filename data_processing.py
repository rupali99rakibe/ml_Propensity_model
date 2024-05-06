# import libraries
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def clean_data(df):
    # data cleaning
    # Replace method for "unknown" variable in ['job', 'education', 'contact']
    df["job"].replace(["unknown"],df["job"].mode(),inplace = True)
    df["education"].replace(["unknown"],df["education"].mode(),inplace = True)
    df["contact"].replace(["unknown"],df["contact"].mode(),inplace = True)
    # remote irrelevant columns
    data = df.drop(['month','day'],axis=1)
    
    # Label encoding
    le = LabelEncoder()
    data['job'] = le.fit_transform(data['job'])
    data['marital'] = le.fit_transform(data['marital'])
    data['education'] = le.fit_transform(data['education'])
    data['default'] = le.fit_transform(data['default'])
    data['housing'] = le.fit_transform(data['housing'])
    data['loan'] = le.fit_transform(data['loan'])
    data['contact'] = le.fit_transform(data['contact'])
    data['poutcome'] = le.fit_transform(data['poutcome'])
    data['deposit'] = le.fit_transform(data['deposit'])
    
    # Standard features
    features = data.drop("deposit",axis = 1)
    target = data['deposit']
    features_num = features.columns
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features))
    features.columns = features_num
    
    # Export clean and processed data
    final_data = pd.concat([features,target], axis=1)
    final_data.to_csv("final_version.csv",index=False)
    print("Cleansed data stored in a local directory named as final_version.csv")
# test a function
if __name__ == "__main__":
    # Read data
    path = 'Data_Science_Project/propensity model/final_version.csv'
    # Load the data frame
    # df = pd.read_csv(path, sep=';')
    clean_data(df) 