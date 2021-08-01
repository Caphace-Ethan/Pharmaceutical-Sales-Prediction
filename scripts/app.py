import numpy as np
from numpy import mean, std
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings

# Import packages for logging
import logging
import logging.handlers
import os

import mlflow
import mlflow.sklearn

dataset_path = './data/'
dataset_version = 'v2'


def load_logging():
    handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "../logs/deep_learning.log"))
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    root.addHandler(handler)
    logging.info("Testing Loggings") 
    try:
        exit(main())
    except Exception:
        logging.exception("Exception in main()")
        exit(1)


def loadData_and_scalling():
    warnings.filterwarnings('ignore')

    train_store_data = pd.read_csv(dataset_path+'train_store_data.csv')
    train_data = pd.read_csv(dataset_path+'train.csv')
    test_store_data = pd.read_csv(dataset_path+'test_store_data.csv')
    df = pd.DataFrame()
    df['Sales'] = train_data['Sales']
    scaler = StandardScaler()

    scale = scaler.fit(df[['Sales']]) 
    # transform the training data column
    df['Sales'] = scale.transform(df[['Sales']])
    train_store_data['Sales'] = df['Sales']

    return train_data, train_store_data, test_store_data


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
    print("Loading logging, and Dataset")
    load_logging()
    try:

        train_data, train_store_data, test_store_data = loadData_and_scalling()
        logging.info(f"Dataset loaded and scalled successfully")
    except Exception as e:
        print(e)
        logging.exception(f" Exception occured in reading, and Scalling dataset, {e}")

    # Splitting the Dataset to train, test and Validation in ratio of 70%, 20%, and 10%
    y_target = train_store_data.Sales
    x_features =  train_store_data.drop(columns=['Sales'], axis=1)
    
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