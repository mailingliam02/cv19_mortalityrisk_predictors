# -*- coding: utf-8 -*-
"""
Created on Sun May  2 10:07:28 2021

@author: Percy
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection
from tensorflow import keras
import pandas as pd
import pickle

path = "C:\\PythonScripts\\Cv19\\cleansed_data1.csv"
data = pd.read_csv(path)
y = data.pop('outcome')
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, y, test_size=0.33, random_state = 5)

rf_model = "cv19_rf_best.pkl"
with open(rf_model, 'rb') as file:
    rf = pickle.load(file)

svm_model = "cv19_svm.pkl"
with open(svm_model, 'rb') as file:
    svm = pickle.load(file)

nn = keras.models.load_model("nn_class")

def classify_num(list1):
    for i in range(len(list1)):
        if list1[i][0] >= 0.5:
            list1[i][0] = 1
        else:
            list1[i][0] = 0
    return list1


def evaluate(data, y_test, model):
    # y_test = data.pop("income")
    # print(data.shape[1])
    y_pred = model.predict(data)
    if model == nn:
      y_pred = classify_num(y_pred)
    list1 = []
    list1.append(metrics.accuracy_score(y_test,y_pred))
    list1.append(metrics.precision_score(y_test,y_pred))
    list1.append(metrics.recall_score(y_test,y_pred))
    list1.append(metrics.roc_auc_score(y_test,y_pred))
    print(str(model)+": ",list1)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)  
    # y_pred_round = classify_num(y_pred)
    C = metrics.confusion_matrix(y_test, y_pred)
    print(C)
    # di = (C[1][1]+C[0][1])/y_test.shape[0]
    plt.scatter(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate') 
    return C
    # return di

# C1 = evaluate(x_test, y_test, rf)
C2 = evaluate(x_test, y_test, svm)
# C3 = evaluate(x_test, y_test, nn)


