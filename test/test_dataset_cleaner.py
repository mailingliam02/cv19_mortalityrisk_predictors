# -*- coding: utf-8 -*-
"""
Test Class
Citations:
Muller, David. "How To Use unittest to Write a Test Case for a Function in Python." 
    Digital Ocean, 30 Sept. 2020, www.digitalocean.com/community/tutorials/
    how-to-use-unittest-to-write-a-test-case-for-a-function-in-python. 
    
"""
import os
import unittest
import numpy as np
import pandas as pd
from src.clean_covid_dataset import Dataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        path = ".\\test\\fake_data.csv"
        save_path = ".\\test\\clean_fake_data.csv"
        self.dataset_cleaner = Dataset(path, save_path)
    
    def tearDown(self):
        #https://careerkarma.com/blog/python-check-if-file-exists/
        file_exist = os.path.isfile(".\\test\\clean_fake_data.csv")
        if file_exist:
            os.remove(".\\test\\clean_fake_data.csv")
        
    def test_remove_no_outcome_success(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death",np.nan,"discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        actual = self.dataset_cleaner._Dataset__remove_no_outcome(preclean_data).reset_index(drop = True)
        test_case_clean = {"age": ["10", "20"], 
                     "sex": ["male", "female"], 
                     "latitude": [1.1, 3.3], 
                     "longitude": [1.3, 1.3],
                     "date_onset_symptoms": ["11.01.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "25.02.20"], 
                     "symptoms": ["cough, respiratory distress","fever"], 
                     "chronic_disease": ["diabetes", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20","25.02.20"], 
                     "outcome": ["death","discharged"], 
                     "admin1": ["True","False"]}
        preclean_data1 = pd.DataFrame(data = test_case_clean, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)
        
    def test_remove_no_outcome_none(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        actual = self.dataset_cleaner._Dataset__remove_no_outcome(preclean_data).reset_index(drop = True)
        test_case_clean = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data1 = pd.DataFrame(data = test_case_clean, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)
        
    def test_remove_no_outcome_all(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": [np.nan,np.nan,np.nan], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        actual = self.dataset_cleaner._Dataset__remove_no_outcome(preclean_data).reset_index(drop = True)
        preclean_data1 = pd.DataFrame(columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected, check_dtype = False)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)
    
    def test_convert_dates_success(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.2020", "24.03.2020", "25.02.2020"], 
                     "date_admission_hospital": ["11.01.2020", "24.03.2020", "25.02.2020"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.2020", "24.03.2020", "25.02.2020"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        actual = self.dataset_cleaner._Dataset__convert_dates(preclean_data, "date_death_or_discharge").reset_index(drop = True)
        test_case_clean = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.2020", "24.03.2020", "25.02.2020"],
                     "date_admission_hospital": ["11.01.2020", "24.03.2020", "25.02.2020"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge_days": [375, 448, 420], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data1 = pd.DataFrame(data = test_case_clean, columns = ["age", "sex", "latitude", "longitude",
                                  "date_onset_symptoms", "date_admission_hospital", 
                                  "symptoms", "chronic_disease", "date_death_or_discharge_days", 
                                  "outcome"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)

    def test_convert_dates_exception(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.2020", "24.03.2020", "25.02.2020"], 
                     "date_admission_hospital": ["11.01.2020", "24.03.2020", "25.02.2020"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.2020", "25.02.2020"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        with self.assertRaises(ValueError) as exception_context:
            actual = self.dataset_cleaner._Dataset__convert_dates(preclean_data, "date_death_or_discharge").reset_index(drop = True)
        self.assertEqual(str(exception_context.exception),"time data '11.01.20' does not match format '%d.%m.%Y' (match)")
        
    def test_convert_dates_negative_exception(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.2020", "24.03.2020", "25.02.2020"], 
                     "date_admission_hospital": ["11.01.2020", "24.03.2020", "25.02.2020"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.2018", "24.03.2020", "25.02.2020"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        with self.assertRaises(ValueError) as exception_context:
            actual = self.dataset_cleaner._Dataset__convert_dates(preclean_data, "date_death_or_discharge").reset_index(drop = True)
        self.assertEqual(str(exception_context.exception),"Date can not be from before 1st of January, 2019")

    def test_sex_encoder_success(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        actual = self.dataset_cleaner._Dataset__sex_standardize(preclean_data).reset_index(drop = True)
        test_case_clean = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"],
                     "sex_encoded": [0, 0, 1], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data1 = pd.DataFrame(data = test_case_clean, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1","sex_encoded"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)
        
    def test_sex_encoder_exception(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["None", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        with self.assertRaises(ValueError) as exception_context:
            actual = self.dataset_cleaner._Dataset__sex_standardize(preclean_data).reset_index(drop = True)
        self.assertEqual(str(exception_context.exception),"Biological gender must be either female or male")

    def test_symptom_encoder_success(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, acute respiratory distress", "headache", "hypoxia"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        actual = self.dataset_cleaner._Dataset__symptom_standardize(preclean_data).reset_index(drop = True)
        test_case_clean = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"],
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": [4, 1, 3], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data1 = pd.DataFrame(data = test_case_clean, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)
        
    def test_symptom_encoder_missing(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, acute respiratory distress", np.nan, "hypoxia"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        actual = self.dataset_cleaner._Dataset__symptom_standardize(preclean_data).reset_index(drop = True)
        test_case_clean = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"],
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": [4, -1, 3], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data1 = pd.DataFrame(data = test_case_clean, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)
    
    def test_symptom_encoder_exception(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["Not a Symptom, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        with self.assertRaises(ValueError) as exception_context:
            actual = self.dataset_cleaner._Dataset__symptom_standardize(preclean_data).reset_index(drop = True)
        self.assertEqual(str(exception_context.exception),"Not a valid symptom")
        
    def test_chronic_disease_encoder_success(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, acute respiratory distress", "headache", "hypoxia"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        actual = self.dataset_cleaner._Dataset__chronic_disease_standardize(preclean_data).reset_index(drop = True)
        test_case_clean = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"],
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, acute respiratory distress", "headache", "hypoxia"],
                     "chronic_disease": [[0,0,0,1], [0,1,0,0], [0,0,1,0]],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"],
                     "diabetes": [1, 0, 0], 
                     "hypertension": [0,0,1], 
                     "Severe Underlying": [0,1,0], 
                     "Other Underlying": [0,0,0]}
        preclean_data1 = pd.DataFrame(data = test_case_clean, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1", 
                              "Other Underlying", "Severe Underlying", 
                              "hypertension", "diabetes"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)
        
    def test_chronic_disease_encoder_empty(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, acute respiratory distress", "headache", "hypoxia"], 
                     "chronic_disease": [np.nan, "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        actual = self.dataset_cleaner._Dataset__chronic_disease_standardize(preclean_data).reset_index(drop = True)
        test_case_clean = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"],
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, acute respiratory distress", "headache", "hypoxia"],
                     "chronic_disease": [[-1,-1,-1,-1], [0,1,0,0], [0,0,1,0]],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"],
                     "diabetes": [-1, 0, 0], 
                     "hypertension": [-1,0,1], 
                     "Severe Underlying": [-1,1,0], 
                     "Other Underlying": [-1,0,0]}
        preclean_data1 = pd.DataFrame(data = test_case_clean, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1", 
                              "Other Underlying", "Severe Underlying", 
                              "hypertension", "diabetes"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)
        
    def test_chronic_disease_encoder_number(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, acute respiratory distress", "headache", "hypoxia"], 
                     "chronic_disease": [1, "COPD", "hypertension"],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        actual = self.dataset_cleaner._Dataset__chronic_disease_standardize(preclean_data).reset_index(drop = True)
        test_case_clean = {"age": ["10", "15", "20"], 
                     "sex": ["male", "male", "female"],
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, acute respiratory distress", "headache", "hypoxia"],
                     "chronic_disease": [[-1,-1,-1,-1], [0,1,0,0], [0,0,1,0]],        
                     "date_death_or_discharge": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"],
                     "diabetes": [-1, 0, 0], 
                     "hypertension": [-1,0,1], 
                     "Severe Underlying": [-1,1,0], 
                     "Other Underlying": [-1,0,0]}
        preclean_data1 = pd.DataFrame(data = test_case_clean, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1", 
                              "Other Underlying", "Severe Underlying", 
                              "hypertension", "diabetes"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)

    def test_outcome_standardize_success(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex_encoded": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms_days": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital_days": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge_days": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"],
                     "diabetes": [-1, 0, 0], 
                     "hypertension": [-1,0,1], 
                     "Severe Underlying": [-1,1,0], 
                     "Other Underlying": [-1,0,0]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex_encoded", "latitude", "longitude",
                              "date_onset_symptoms_days", "date_admission_hospital_days", 
                              "date_death_or_discharge_days", "outcome", "symptoms", 
                              "diabetes", "hypertension", "Severe Underlying", 
                              "Other Underlying"])
        actual = self.dataset_cleaner._Dataset__outcome_standardize(preclean_data).reset_index(drop = True)
        test_case_clean = {"age": ["10", "15", "20"], 
                     "sex_encoded": ["male", "male", "female"],
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms_days": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital_days": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "outcome": [1,0,0], 
                     "admin1": ["True","False","False"],
                     "diabetes": [-1, 0, 0], 
                     "hypertension": [-1,0,1], 
                     "Severe Underlying": [-1,1,0], 
                     "Other Underlying": [-1,0,0]}
        preclean_data1 = pd.DataFrame(data = test_case_clean, columns = ["age", "sex_encoded", "latitude", "longitude",
                              "date_onset_symptoms_days", "date_admission_hospital_days", 
                              "outcome", "symptoms", "diabetes", "hypertension", 
                              "Severe Underlying", "Other Underlying"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)
        
    def test_outcome_standardize_exception(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex_encoded": ["male", "male", "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms_days": ["11.01.20", "24.03.20", "25.02.20"], 
                     "date_admission_hospital_days": ["11.01.20", "24.03.20", "25.02.20"], 
                     "symptoms": ["cough, respiratory distress", "headache", "fever"], 
                     "chronic_disease": ["diabetes", "COPD", "hypertension"],        
                     "date_death_or_discharge_days": ["11.01.20", "24.03.20", "25.02.20"], 
                     "outcome": ["death","Free","discharged"], 
                     "admin1": ["True","False","False"],
                     "diabetes": [-1, 0, 0], 
                     "hypertension": [-1,0,1], 
                     "Severe Underlying": [-1,1,0], 
                     "Other Underlying": [-1,0,0]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex_encoded", "latitude", "longitude",
                              "date_onset_symptoms_days", "date_admission_hospital_days", 
                              "date_death_or_discharge_days", "outcome", "symptoms", 
                              "diabetes", "hypertension", "Severe Underlying", 
                              "Other Underlying"])
        with self.assertRaises(ValueError) as exception_context:
            actual = self.dataset_cleaner._Dataset__outcome_standardize(preclean_data).reset_index(drop = True)
        self.assertEqual(str(exception_context.exception),"Invalid Outcome Entry")
        
    def test_normalize_success(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex_encoded": [0, -1, 1], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms_days": [357, 10, 47], 
                     "date_admission_hospital_days": [357, 10, 47],
                     "symptoms": [4, 1, -1],       
                     "outcome": [1,0,0], 
                     "diabetes": [-1, 0, 0], 
                     "hypertension": [-1,0,1], 
                     "Severe Underlying": [-1,1,0], 
                     "Other Underlying": [-1,0,0]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex_encoded", "latitude", "longitude",
                      "date_onset_symptoms_days", "date_admission_hospital_days", 
                      "outcome", "symptoms", 
                      "diabetes", "hypertension", "Severe Underlying", 
                      "Other Underlying"])
        actual = self.dataset_cleaner._Dataset__normalize(preclean_data).reset_index(drop = True)
        test_case_clean = {"age": [0, 0.5, 1], 
                     "sex_encoded": [0.5, 0, 1], 
                     "latitude": [0, 0.5, 1], 
                     "longitude": [0, 1, 0],
                     "date_onset_symptoms_days": [1, 0, 0.10662824207492795], 
                     "date_admission_hospital_days": [1, 0, 0.10662824207492795],
                     "symptoms": [1, 0.4, 0],       
                     "outcome": [1,0,0], 
                     "diabetes": [0, 1, 1], 
                     "hypertension": [0,0.5,1], 
                     "Severe Underlying": [0,1,0.5], 
                     "Other Underlying": [0,1,1]}
        preclean_data1 = pd.DataFrame(data = test_case_clean, columns = ["age", "sex_encoded", "latitude", "longitude",
                      "date_onset_symptoms_days", "date_admission_hospital_days", 
                      "outcome", "symptoms", 
                      "diabetes", "hypertension", "Severe Underlying", 
                      "Other Underlying"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected, check_dtype = False)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)
        
    def test_normalize_exception(self):    
        test_case = {"age": ["10", "15", "20"], 
                     "sex_encoded": [0, -1, 1], 
                     "latitude": ["4", 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms_days": [357, 10, 47], 
                     "date_admission_hospital_days": [357, 10, 47],
                     "symptoms": [4, 1, -1],       
                     "outcome": [1,0,0], 
                     "diabetes": [-1, 0, 0], 
                     "hypertension": [-1,0,1], 
                     "Severe Underlying": [-1,1,0], 
                     "Other Underlying": [-1,0,0]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex_encoded", "latitude", "longitude",
                      "date_onset_symptoms_days", "date_admission_hospital_days", 
                      "outcome", "symptoms", 
                      "diabetes", "hypertension", "Severe Underlying", 
                      "Other Underlying"])
        with self.assertRaises(TypeError) as exception_context:
            actual = self.dataset_cleaner._Dataset__normalize(preclean_data).reset_index(drop = True)
        self.assertEqual(str(exception_context.exception),"All columns must be numerical to normalize")
        
    def test_clean_and_save_success(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", np.nan, "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["24.12.2019", "11.01.2019", "17.02.2019"], 
                     "date_admission_hospital": ["24.12.2019", "11.01.2019", "17.02.2019"],
                     "symptoms": ["cough, acute respiratory failure", "headache", np.nan], 
                     "chronic_disease": [np.nan, "COPD", "hypertension"],        
                     "date_death_or_discharge": ["24.12.2019", "11.01.2019", "17.02.2019"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        self.dataset_cleaner.data_pc = preclean_data
        actual = self.dataset_cleaner.clean_and_save().reset_index(drop = True)
        test_case_clean = {"age": [0, 0.5, 1], 
                     "sex_encoded": [0.5, 0, 1], 
                     "latitude": [0, 0.5, 1], 
                     "longitude": [0, 1, 0],
                     "date_onset_symptoms_days": [1, 0, 0.10662824207492795], 
                     "date_admission_hospital_days": [1, 0, 0.10662824207492795],
                     "symptoms": [1, 0.4, 0],       
                     "outcome": [1,0,0], 
                     "diabetes": [0, 1, 1], 
                     "hypertension": [0,0.5,1], 
                     "Severe Underlying": [0,1,0.5], 
                     "Other Underlying": [0,1,1]}
        preclean_data1 = pd.DataFrame(data = test_case_clean, columns = ["age", "sex_encoded", "latitude", "longitude",
                      "date_onset_symptoms_days", "date_admission_hospital_days", 
                      "outcome", "symptoms", 
                      "diabetes", "hypertension", "Severe Underlying", 
                      "Other Underlying"])
        expected = preclean_data1.reset_index(drop = True)
        #Should raise error
        pd.testing.assert_frame_equal(actual, expected, check_dtype = False)
        #If above does not raise error, means they are equivalent. Hence Assert True
        self.assertEqual(1,1)
        
    def test_clean_and_save_checkSave_success(self):
        test_case = {"age": ["10", "15", "20"], 
                     "sex": ["male", np.nan, "female"], 
                     "latitude": [1.1, 2.2, 3.3], 
                     "longitude": [1.3, 2.6, 1.3],
                     "date_onset_symptoms": ["24.12.2019", "11.01.2019", "17.02.2019"], 
                     "date_admission_hospital": ["24.12.2019", "11.01.2019", "17.02.2019"],
                     "symptoms": ["cough, acute respiratory failure", "headache", np.nan], 
                     "chronic_disease": [np.nan, "COPD", "hypertension"],        
                     "date_death_or_discharge": ["24.12.2019", "11.01.2019", "17.02.2019"], 
                     "outcome": ["death","discharged","discharged"], 
                     "admin1": ["True","False","False"]}
        preclean_data = pd.DataFrame(data = test_case, columns = ["age", "sex", "latitude", 
                              "longitude", "date_onset_symptoms", 
                              "date_admission_hospital", "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"])
        self.dataset_cleaner.data_pc = preclean_data
        actual = self.dataset_cleaner.clean_and_save().reset_index(drop = True)
        file_exist = os.path.isfile(".\\test\\clean_fake_data.csv")
        if file_exist:
            self.assertEqual(1,1)
        else:
            raise Exception("File has not been saved")
            self.assertEqual(0,1)
    
        
        
        