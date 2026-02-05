# src/utils.py
# -------------------------------------------------
# This module provides utility functions for the application.
# It includes functions for common tasks that can be reused across different modules.   
# =================================================


# =============================================
# Importing the Necessary Libraries
# ---------------------------------------------
# OS Library for interacting with the operating system
# SYS Library for system-specific parameters and functions
# Numpy for numerical operations
# Pandas for data manipulation and analysis
# Dill for serializing and deserializing Python objects
# Pickle for object serialization
# Sklearn's metrics for evaluating model performance
# Sklearn's model_selection for hyperparameter tuning
# src.exception for custom exception handling
# =============================================
import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


# =============================================
# save_object Function
# ---------------------------------------------
# This function saves a Python object to a specified file path using pickle serialization.
# It creates the necessary directories if they do not exist.
# =============================================
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

# =============================================
# evaluate_models Function
# ---------------------------------------------
# This function evaluates multiple machine learning models using GridSearchCV for hyperparameter tuning.
# It returns a report of the R2 scores for each model on the test dataset.
# =============================================    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

# =============================================
# load_object Function
# ---------------------------------------------
# This function loads a Python object from a specified file path using pickle deserialization.
# =============================================    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
