# -*- coding: utf-8 -*-
"""
Code to train, evaluate and save the SVM trained on the cleaned datset. 
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
import numpy as np
from sklearn import metrics, model_selection, svm
import pandas as pd
from imblearn.combine import SMOTEENN
import pickle
#Set
path = "build\\cleandata.csv"
grid_search = True
def train_and_save(path, grid_search):
    data = pd.read_csv(path)
    y = data.pop('outcome')
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, y, test_size=0.33, random_state = 0)
    x_train, y_train =  SMOTEENN().fit_resample(x_train, y_train)
    if grid_search:
        C_space = np.logspace(-3, 3, 5)
        gamma_space = np.logspace(-3, 3, 5)
        search_field = dict(gamma=gamma_space, C=C_space)
        model_svm = svm.SVC(C = 1000, gamma = 1000)
        model = model_selection.GridSearchCV(model_svm, search_field, verbose = 2)
        model = model_svm.fit(x_train,y_train)
        list1 = []
        list1.append(metrics.accuracy_score(y_test,model.predict(x_test)))
        list1.append(metrics.precision_score(y_test,model.predict(x_test)))
        list1.append(metrics.recall_score(y_test,model.predict(x_test)))
        list1.append(metrics.roc_auc_score(y_test,model.predict(x_test)))
        with open('holder.txt','w') as f:
            f.write(str(list1))
        ##Uncomment to see the parameters which achieved the best accuracy
        # print((model.best_params_, model.best_score_))
    else:
        model_svm = svm.SVC(C = 1000, gamma = 1000)
        model = model_svm.fit(x_train,y_train)
        list1 = []
        list1.append(metrics.accuracy_score(y_test,model.predict(x_test)))
        list1.append(metrics.precision_score(y_test,model.predict(x_test)))
        list1.append(metrics.recall_score(y_test,model.predict(x_test)))
        list1.append(metrics.roc_auc_score(y_test,model.predict(x_test)))
        print(list1)  
    file_model = "cv19_svm.pkl"
    with open(file_model, 'wb') as file:
        pickle.dump(model, file)
    return

train_and_save(path,grid_search)



