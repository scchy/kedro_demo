# train_func_node.py 
# python3
# Create date: 2022-09-05
# Author: Scc_hy
# Func: 模型训练
# ===============================================================================
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def split_data(data, parameters):
    X = data[parameters["features"]]
    y = data[parameters["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train.values.ravel())
    return regressor
