# -*- coding: utf-8 -*-
"""
Code to train, evaluate and save the Random Forest trained on the cleaned datset. 
Use the cleaned dataset (from CV19_data_cleaner.py). If you want to run a gridsearch,
set the grid search bool to true

Citations:
Pandas:  McKinney, Proceedings of the 9th Python in Science Conference, Volume 445, 2010.
sklearn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
SMOTEENN: Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning, 
            Guillaume et al., JMLR 18 (17), pp. 1-5, 2017.
Numpy: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. 
            Nature 585, 357–362 (2020). DOI: 0.1038/s41586-020-2649-2
"""
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from sklearn import metrics, model_selection
import pandas as pd
from imblearn.combine import SMOTEENN
#To Set:
path = "..\\build\\cleandata.csv"
grid_search_bool = True

def train_and_save_rf(path, grid_search_bool):
    data = pd.read_csv(path)
    y = data.pop('outcome')
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, y, test_size=0.15, random_state = 0) #Changed from 0.33
    x_train, y_train =  SMOTEENN().fit_resample(x_train, y_train)
    if grid_search_bool:
        max_depth_range = np.logspace(1, 3, 5, dtype = int)
        minsamplesplit = np.logspace(1, 2, 5, dtype = int)
        max_leaf_range = np.logspace(1, 3, 5, dtype = int)
        minsampleleaf = np.logspace(1, 2, 5, dtype = int)
        search_field = dict(min_samples_split= minsamplesplit, min_samples_leaf = minsampleleaf, max_depth = max_depth_range, max_leaf_nodes = max_leaf_range)
        model_rf = RandomForestClassifier(max_depth = 100, max_leaf_nodes = 1000, min_samples_leaf = 10, min_samples_split = 17)
        model = model_selection.GridSearchCV(model_rf, search_field, verbose = 2,scoring = "roc_auc") #Will help find the right values!
        model = model.fit(x_train,y_train)
        list1 = []
        list1.append(metrics.accuracy_score(y_test,model.predict(x_test)))
        list1.append(metrics.precision_score(y_test,model.predict(x_test)))
        list1.append(metrics.recall_score(y_test,model.predict(x_test)))
        list1.append(metrics.roc_auc_score(y_test,model.predict(x_test)))
        with open('holder.txt','a') as f:
            f.write('\n')
            f.write(str(list1))
        #Uncomment the below to see the parameters which return the highest accuracy
        # print("Best RF:",(model.best_params_, model.best_score_)) #5-fold cross validation is used
    else:
        model_rf = RandomForestClassifier(max_depth = 100, max_leaf_nodes = 1000, min_samples_leaf = 10, min_samples_split = 17)
        model = model_rf.fit(x_train,y_train)
        list1 = []
        list1.append(metrics.accuracy_score(y_test,model.predict(x_test)))
        list1.append(metrics.precision_score(y_test,model.predict(x_test)))
        list1.append(metrics.recall_score(y_test,model.predict(x_test)))
        list1.append(metrics.roc_auc_score(y_test,model.predict(x_test)))
        print(list1)
    file_model = "cv19_rf_best.pkl"
    with open(file_model, 'wb') as file:
        pickle.dump(model, file)
    return

train_and_save_rf(path, grid_search_bool)




