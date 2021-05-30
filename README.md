# Predicting the Mortality Risk for Covid 19
## Introduction
This is a repository containing the python files used in the "Using Machine Learning to Predict Patient mortality" paper (see docs for full paper pdf). The purpose of the project was to compare how a Random Forest, Support Vector Machine and Neural Network performed in predicting patient mortality at the point of admission to a hospital on retrospective data. They were trained on the dataset compiled by Xu et al. in their "Epidemiological data from the COVID-19 outbreak, real-time case information" paper. The dataset, which contains mixed categorical and numerical data was converted into a numerical dataset and applied into the above modules. Due to the large imbalance in the dataset, a synthetic minority oversampling technique with editted nearest neighbor was used to help balance the data.
## Results
The Random Forest was found to be the best predictor with an accuracy of 95%, although it is noted that none of the models were particularly effective at predicting the death of a patient. This is attributed to the sampling technique used. See the paper for details about implementation choices and results.
Table containing performance metrics for each of the different classifiers:

![tablef](https://user-images.githubusercontent.com/71287923/120115746-6983f980-c185-11eb-9aa9-4ecc8ac60b2e.PNG)


Plot of recall vs precision for each of the five trials of the classifier:
![precvsrecall1](https://user-images.githubusercontent.com/71287923/120115714-4a856780-c185-11eb-85f2-562366255b4a.png)
