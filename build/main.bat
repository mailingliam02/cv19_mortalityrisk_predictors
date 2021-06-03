@echo off
cd ..
echo Testing successful installation
python -m test.test_dataset_cleaner && echo Test Successful && python src\clean_covid_dataset.py && echo Dataset cleaned, training SVM && python src\svm.py && echo SVM trained, training RF && python src\rf.py && echo RF trained, training NN && python src\nn.py && Scoring... && python src\score.py
@pause
