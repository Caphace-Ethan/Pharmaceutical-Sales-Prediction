import numpy as np
from numpy import mean, std
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn import preprocessing

# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings

import mlflow
import mlflow.sklearn

dataset_path = './data/'
dataset_version = 'v1'


def loadData():
    warnings.filterwarnings('ignore')

    df_browser1 = pd.read_csv(dataset_path+'browser1_dataset.csv')
    df_browser2 = pd.read_csv(dataset_path+'browser2_dataset.csv')
    df_platfromOs1 = pd.read_csv(dataset_path+'platform_os1_dataset.csv')
    df_platfromOs2 = pd.read_csv(dataset_path+'platform_os2_dataset.csv')

    return df_browser1, df_browser2, df_platfromOs1, df_platfromOs2

# Data Preparation
def encode_labels(dataframe):

    experiment_encoder = preprocessing.LabelEncoder()
    date_encoder = preprocessing.LabelEncoder()
    hour_encoder = preprocessing.LabelEncoder()
    device_encoder = preprocessing.LabelEncoder()
    browser_encoder = preprocessing.LabelEncoder()
    platform_encoder = preprocessing.LabelEncoder()
    aware_encoder = preprocessing.LabelEncoder()
    
    
    dataframe['date'] = date_encoder.fit_transform(dataframe['date'])
    dataframe['hour'] = hour_encoder.fit_transform(dataframe['hour'])
    dataframe['device_make'] = device_encoder.fit_transform(dataframe['device_make'])
    dataframe['browser'] = browser_encoder.fit_transform(dataframe['browser'])
    dataframe['experiment'] = experiment_encoder.fit_transform(dataframe['experiment'])
    dataframe['platform_os'] = platform_encoder.fit_transform(dataframe['platform_os'])
    dataframe['aware'] = aware_encoder.fit_transform(dataframe['aware'])
    
    return dataframe 

#create feature and target column
def dataset_features(df):
    df1 = encode_labels(df)
    feature_col =["experiment", "hour", "date", "device_make","browser","platform_os"]
    features_X = df1[feature_col]
    target_y = df1["aware"]

    return features_X, target_y

# browser1_df_features, browser1_df_target = dataset_features(browser1_dataset)
# browser1_df_features

def train_test_val_split(X, Y, split=(0.2, 0.1), shuffle=True):
    

    assert len(X) == len(Y), 'The length of X and Y must be consistent.'
    X_train, X_test_val, y_train, Y_test_val = train_test_split(X, Y, test_size=(split[0]+split[1]), shuffle=shuffle)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, Y_test_val, 
        test_size=split[1], shuffle=False)
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

#fit model
def fit_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    return model

#predict
def model_predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


if __name__ == "__main__":
    mlflow.set_experiment(experiment_name='Exp01-Logistic Regression')
    print("Loading Dataset")
    # mlflow.sklearn.log_version("v1")
    df_browser1, df_browser2, df_platfromOs1, df_platfromOs2 = loadData()

    # Splitting the Dataset to train, test and Validation in ratio of 70%, 20%, and 10%

    browser1_X, browser1_Y = dataset_features(df_browser1)
    browser2_X, browser2_Y = dataset_features(df_browser2)
    platform1_X, platform1_Y = dataset_features(df_platfromOs1)
    platform2_X, platform2_Y = dataset_features(df_platfromOs2)

    (X1_train, y1_train), (X1_test, y1_test), (X1_val, y1_val)=train_test_val_split(browser1_X, browser1_Y)
    (X2_train, y2_train), (X2_test, y2_test), (X2_val, y2_val)=train_test_val_split(browser2_X, browser2_Y)
    (X3_train, y3_train), (X3_test, y3_test), (X3_val, y3_val)=train_test_val_split(platform1_X, platform1_Y)
    (X4_train, y4_train), (X4_test, y4_test), (X4_val, y4_val)=train_test_val_split(platform2_X, platform2_Y)
    
    # Modelling 

    # create model
    model = LogisticRegression()
    mlflow.sklearn.log_model(model, "model")
    # Test for All datasets
    target1_predictions = model_predict(fit_model(model,X1_train, y1_train), X1_test)
    target2_predictions = model_predict(fit_model(model,X2_train, y2_train), X2_test)
    target3_predictions = model_predict(fit_model(model,X3_train, y3_train), X3_test)
    target4_predictions = model_predict(fit_model(model,X4_train, y4_train), X4_test)

    # evaluating the model
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    scores1 = cross_val_score(model, X1_train, y1_train, scoring='accuracy', cv=cv, n_jobs=-1)
    scores2 = cross_val_score(model, X2_train, y2_train, scoring='accuracy', cv=cv, n_jobs=-1)
    scores3 = cross_val_score(model, X3_train, y3_train, scoring='accuracy', cv=cv, n_jobs=-1)
    scores4 = cross_val_score(model, X4_train, y4_train, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy for browser1 dataset (Chrome Mobile): %.3f (%.3f)' % (mean(scores1), std(scores1)))
    print('Accuracy for browser1 dataset (Chrome Mobile WebView): %.3f (%.3f)' % (mean(scores2), std(scores2)))
    print('Accuracy for platformOs1 dataset (6): %.3f (%.3f)' % (mean(scores3), std(scores3)))
    print('Accuracy for platformOs2 dataset (5): %.3f (%.3f)' % (mean(scores4), std(scores4)))


    mlflow.log_metric("Accuracy Dataset1", mean(scores1))
    mlflow.log_metric("Accuracy Dataset2", mean(scores2))
    mlflow.log_metric("Accuracy Dataset3", mean(scores3))
    mlflow.log_metric("Accuracy Dataset4", mean(scores4))

    cnf1_matrix = metrics.confusion_matrix(y1_test, target1_predictions)
    cnf2_matrix = metrics.confusion_matrix(y2_test, target2_predictions)
    cnf3_matrix = metrics.confusion_matrix(y3_test, target3_predictions)
    cnf4_matrix = metrics.confusion_matrix(y4_test, target4_predictions)

    cnf_matrices = [cnf1_matrix, cnf2_matrix, cnf3_matrix, cnf4_matrix]
    i=1
    for cnf_matrix in cnf_matrices:
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        plt.title(f"Figure {i}, Confusion matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        i += 1
    # mlflow.log_metric("score", score)
    # mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)