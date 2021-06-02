@echo off
python ..\test\test_dataset_cleaner.py && python ..\src\clean_covid_dataset.py && python ..\src\svm.py && python ..\src\rf.py && python ..\src\nn.py && python ..\src\score.py
@pause
