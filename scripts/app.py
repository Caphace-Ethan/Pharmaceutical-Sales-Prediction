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
def fit_model(x_train, y_train):
    #training using cross validation set
    regressor_validation=RandomForestRegressor(n_estimators=128, 
                                criterion='mse', 
                                max_depth=20, 
                                min_samples_split=10, 
                                min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, 
                                max_features='auto', 
                                max_leaf_nodes=None, 
                                min_impurity_decrease=0.0, 
                                min_impurity_split=None, 
                                bootstrap=True, 
                                oob_score=False,
                                n_jobs=4, #setting n_jobs to 4 makes sure you're using the full potential of the machine you're running the training on
                                random_state=35, 
                                verbose=0, 
                                warm_start=False)
    model_test=regressor_validation.fit(x_train,y_train)

    return model_test

# Applying loss function to see if our model is quite correct
def rmspe(y, y_pred):
    rmspe = np.sqrt(np.mean( (y - y_pred)**2 ))
    return rmspe


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
    
    try: 
        x_train, x_train_test, y_train, y_train_test = train_test_split(x_features, y_target, test_size=0.20, random_state=15)

    except Exception as e:
        print(e)
        logging.debug(f"Exception occured in separating dataset into x & y_training dataset, {e}")

    # Modelling 
    model = fit_model(x_train, y_train)
    
    mlflow.sklearn.log_model(model, "model")
    # Test for All datasets
    Y_pred = model.predict(x_train_test) 
    plt.hist(Y_pred)

    error=rmspe(y_train_test,Y_pred)
    # evaluating the model

    print('Error associated with our Model is: %.3f ' % error)


    mlflow.log_metric("Error Associated with the Model is", error)
    
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)