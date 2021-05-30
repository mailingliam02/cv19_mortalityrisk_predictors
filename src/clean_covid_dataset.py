# -*- coding: utf-8 -*-
"""
To run code, need to specify two paths: the path variable, where the original data is located,
and the save path, which specifies where to save the cleaned data.

Citations:
Numpy: Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. 
            Nature 585, 357–362 (2020). DOI: 0.1038/s41586-020-2649-2
Pandas:  McKinney, Proceedings of the 9th Python in Science Conference, Volume 445, 2010.
"""
import datetime
import math
import numpy as np
import pandas as pd
import sys

input_arguments = str(sys.argv)
path = input_arguments[1]
save_path = input_arguments[2]

class Dataset:
    def __init__(self, path, save_path):
        self.path = path
        self.save_path = save_path
        self.data = pd.read_csv(self.path, low_memory = False)
        self.data_pc = self.data.loc[:,["age", "sex", "latitude", "longitude",
                      "date_onset_symptoms", "date_admission_hospital", 
                      "symptoms", "chronic_disease", "date_death_or_discharge", 
                      "outcome", "admin1"]]

        
    def __remove_no_outcome(self, data_pc):
        """
        Remove data for which there was no listed outcome

        Parameters
        ----------
        data_pc : Pandas Dataframe
            Dataset after being read into pandas and with the relevant titles.

        Returns
        -------
        data_mc : Pandas Dataframe
            Dataset with all entries having outcomes.

        """
        hold_data = data_pc.assign(bool_outcome = pd.isna(data_pc["outcome"]))
        holder_data = hold_data.query('bool_outcome == False')
        data_mc = holder_data.loc[:,["age", "sex", "latitude", "longitude",
                              "date_onset_symptoms", "date_admission_hospital", 
                              "symptoms", "chronic_disease", 
                              "date_death_or_discharge", "outcome", "admin1"]]
        return data_mc

    def __convert_dates(self, data_mc, column):
        """
        Converts a column in DD.MM.YYYY to days since January 1st, 2019.
        Set up to convert date_death_or_discharge first, followed by date_onset
        _symptoms followed by date_admission_hospital

        Parameters
        ----------
        data_mc : Pandas Dataframe
            Dataset with dates.
        column : String
            Title of column with dates (exclusively).

        Returns
        -------
        data_mc : Pandas Dataframe
            Dataset with numerical dates.

        """
        if column == "date_death_or_discharge":
                data_mc.loc[data_mc['date_death_or_discharge'] == '03.22.2020',
                            'date_death_or_discharge'] = "22.03.2020"
        elif column == "date_onset_symptoms":
            def typo_fixer(x): #Needed to fix a typo
                try:
                    if math.isnan(x) != True:
                        if x[0] == "-":
                            x = x[1:]
                except:
                    if len(x) >10:
                        x = x[:10]
                return x
            data_mc.loc[:,"date_onset_symptoms_cleaned"] = data_mc.loc[:,"date_onset_symptoms"].apply(typo_fixer) 
        
        data_mc.loc[:,column+"_d"] = pd.to_datetime(data_mc.loc[:,column], format = "%d.%m.%Y")
        hold_data = data_mc.assign(hold_delta = lambda x: x.loc[:,column+"_d"]-datetime.datetime(2019, 1, 1))
        #https://stackoverflow.com/questions/63294040/pandas-check-if-dataframe-has-negative-value-in-any-column
        if (hold_data.loc[:,"hold_delta"].dt.days < 0).values.any():
            raise ValueError("Date can not be from before 1st of January, 2019")
        hold_data.loc[:,column+"_days"] = hold_data.loc[:,"hold_delta"].dt.days
        if column == "date_death_or_discharge":
            data_mc = hold_data.loc[:,["age", "sex", "latitude", "longitude",
                                  "date_onset_symptoms", "date_admission_hospital", 
                                  "symptoms", "chronic_disease", "date_death_or_discharge_days", 
                                  "outcome"]]
            
        elif column == "date_onset_symptoms":
            data_mc = hold_data.loc[:,["age", "sex", "latitude", "longitude",
                      "date_onset_symptoms_days", "date_admission_hospital", 
                      "symptoms", "chronic_disease", "date_death_or_discharge_days", 
                      "outcome"]]
            
        elif column == "date_admission_hospital":
            data_mc = hold_data.loc[:,["age", "sex", "latitude", "longitude",
                      "date_onset_symptoms_days", "date_admission_hospital_days", 
                      "symptoms", "chronic_disease", "date_death_or_discharge_days", 
                      "outcome"]]
        return data_mc
    
    def __sex_standardize(self,data_mc):
        """
        Converts the binary gender column into numerical. 0 is male, 1 is 
        female

        Parameters
        ----------
        data_mc : Pandas Dataframe
            Dataset with categorical gender.

        Returns
        -------
        data_mc : Pandas Dataframe
            Dataset with numerical gender.

        """
        def sex_encoder(x):
            a = 0
            if x == "female":
                a = 1
            elif x == "male":
                pass
            elif type(x) != str:
                a = -1
            else:
                raise ValueError("Biological gender must be either female or male")
            return a
        data_mc.loc[:,"sex_encoded"] = data_mc.loc[:,"sex"].apply(sex_encoder) 
        return data_mc
        
    def __symptom_standardize(self,data_mc):
        """
        Converts the mixed categorical data into a numerical representation
        of nominal categorical data. See report for details

        Parameters
        ----------
        data_mc : Pandas Dataframe
            Dataset with categorical symptom data.

        Returns
        -------
        data_mc : Pandas Dataframe
            Dataset with numerically encoded symptom data.

        """
        
        def symptoms_standardize(x):
            """
            Standardize so they can be encoded
            """
            if type(x) == str:
                x.strip()
                sym_hold = []
                sym_list = x.split(",")
                for i in range(len(sym_list)):
                    holder = sym_list[i].split(":")
                    for j in range(len(holder)):
                        hold = holder[j].split(";")
                        for k in range(len(hold)):
                            sym_hold.append(hold[k])
                sym_list = sym_hold.copy()
                for i in range(len(sym_list)):
                    sym_list[i] = sym_list[i].strip()
                    if sym_list[i] == "transient fatigue" or sym_list[i] == "fatigure" or sym_list[i] == "somnolence":
                        sym_list[i] = "fatigue"
                    elif sym_list[i] == "dry cough":
                        sym_list[i] = "cough"
                    elif sym_list[i] == "sensation of chill" or sym_list[i] == "cold chills":
                        sym_list[i] = "chills"
                    elif sym_list[i] == "expectoration" or sym_list[i] == "little sputum":
                        sym_list[i] = "sputum"
                    elif sym_list[i] == "coronary artery stenting":
                        sym_list[i] = "coronary heart disease"
                    elif sym_list[i] == "frequent ventricular premature beat (FVPB)":
                        sym_list[i] = "FVPB"
                    elif sym_list[i] == "respiratory stress" or sym_list[i] == "gasp" or sym_list[i] == "shortness of breath" or sym_list[i] == "grasp":
                        sym_list[i] = "dyspnea"
                    elif sym_list[i] == "chest pain": #This one is a bit debatable
                        sym_list[i] = "chest distress"
                    elif sym_list[i] == "body malaise" or sym_list[i] == "malaise":
                        sym_list[i] = 'discomfort'
                    elif sym_list[i] == 'none' or sym_list[i] == 'afebrile':	
                        sym_list[i] = 'asymptomatic'
                    elif sym_list[i] == 'acute kidney injury':	
                        sym_list[i] = 'acute renal failure'
                    elif sym_list[i] == 'heart failure':	
                        sym_list[i] = 'myocardial infarction'
                    elif sym_list[i] == 'congestive heart failure':	
                        sym_list[i] = 'heart failure'
                    elif sym_list[i] == "dyspnea" or sym_list[i] == "difficulty breathing" or sym_list[i] == "respiratory symptoms":
                        sym_list[i] = "dsypnea"
                    elif sym_list[i] == "acute respiratory distress":
                        sym_list[i] = "acute respiratory distress syndrome" 
                    elif sym_list[i] == "myalgia" or sym_list[i] == "myalgias" or sym_list[i] == "mialgia":
                        sym_list[i] = "muscular soreness"
                    elif sym_list[i] == "kidney failure and hypertension":
                        sym_list[i] = 'acute renal failure'
                    elif sym_list[i] == "running nose":
                        sym_list[i] = 'runny nose'
                    elif sym_list[i] == "weak":
                        sym_list[i] = "systemic weakness"
                    elif sym_list[i] == "chest discomfort":
                        sym_list[i] = "chest distress"
                    elif sym_list[i] == "multiple electrolyte imbalance": #Not really what this means...
                        sym_list[i] = "asymptomatic"
                        
                new_x = ""
                for i in range(len(sym_list)):
                    if i == 0 and len(sym_list) == 1:
                        new_x = str(sym_list[i])
                    elif i == 0:
                        new_x = str(sym_list[i]) + ","
                    elif i != len(sym_list)-1:
                        new_x = new_x + " " + str(sym_list[i]) + ","
                    else:
                        new_x = new_x + " " + str(sym_list[i])
            else:
                new_x = x
            return new_x
        
        def sym_encode(x):
            encode = [-1]*5
            symptom = -1
            if type(x) == str:
                encode = [0]*5
                sym_list = x.split(",")
                for i in range(len(sym_list)):
                    sym_list[i] = sym_list[i].strip()
                    if sym_list[i] == 'acute respiratory failure' or sym_list[i] == 'acute respiratory distress syndrome' or sym_list[i] == 'acute renal failure' or sym_list[i] == 'multiple organ failure' or sym_list[i] == 'myocardial infarction' or sym_list[i] == 'heart failure' or sym_list[i] == 'myocardial dysfunction' or sym_list[i] == 'septic shock' or sym_list[i] == 'acute myocardial infarction' or sym_list[i] == 'cardiogenic shock':
                        encode[4] = 1
                    elif sym_list[i] == 'severe pneumonia' or sym_list[i] == 'acute respiratory disease' or sym_list[i] == '' or sym_list[i] == 'hypoxia' or sym_list[i] == 'lesions on chest radiographs' or sym_list[i] == 'Severe' or sym_list[i] == 'severe acute respiratory infection' or sym_list[i] == "severe":
                        encode[3] = 1
                    elif sym_list[i] == 'pneumonia' or sym_list[i] == 'sepsis' or sym_list[i] == 'dsypnea' or sym_list[i] == 'dyspnea' or sym_list[i] == 'chest distress':
                        encode[2] = 1
                    elif sym_list[i] == 'fever' or sym_list[i] == 'cough' or sym_list[i] == 'sore throat' or sym_list[i] == 'discomfort' or sym_list[i] == 'dizziness' or sym_list[i] == 'diarrhea' or sym_list[i] == 'headache' or sym_list[i] == 'emesis' or sym_list[i] == 'runny nose' or sym_list[i] == 'fatigue' or sym_list[i] == 'systemic weakness' or sym_list[i] == 'chills' or sym_list[i] == "colds" or sym_list[i] == "sputum" or sym_list[i] == "muscular soreness":
                        encode[1] = 1
                    elif sym_list[i] == "asymptomatic":
                        encode[0] = 1
                    elif sym_list[i] == "Mild to moderate":
                        encode[2] = 1
                        encode[1] = 1
                    else:
                        raise ValueError("Not a valid symptom")
            if encode[4] == 1:
                symptom = 4
            elif encode[3] == 1:
                symptom = 3
            elif encode[2] == 1:
                symptom = 2
            elif encode[1] == 1:
                symptom = 1
            elif encode[0] == 1:
                symptom = 0   
            return symptom
        
        #Encode Symptoms
        data_mc.loc[:,"symptoms"] = data_mc.loc[:,"symptoms"].apply(symptoms_standardize) 
        data_mc.loc[data_mc['symptoms'] == 'torpid evolution with respiratory distress and severe bronchopneumonia','symptoms'] = "acute respiratory disease, pneumonia"
        data_mc.loc[data_mc['symptoms'] == 'obnubilation','symptoms'] = np.nan
        data_mc.loc[data_mc['symptoms'] == 'primary myelofibrosis','symptoms'] = np.nan
        data_mc.loc[:,"symptoms"] = data_mc.loc[:,"symptoms"].apply(sym_encode) 
        return data_mc
    
    def __chronic_disease_standardize(self, data_mc):
        """
        Converts the mixed categorical data into 4 numerical representation
        of binary categorical data. See report for details

        Parameters
        ----------
        data_mc : Pandas Dataframe
            Dataset with categorical chronic disease data.

        Returns
        -------
        data_mc : Pandas Dataframe
            Dataset with numerically encoded chronic disease data, split into
            4 columns ("diabetes", "hypertension", "Severe Underlying", 
                       "Other Underlying")

        """
        def chronic_disease_standardize(x):        
            if type(x) == str:
                x.strip()
                sym_hold = []
                sym_list = x.split(",")
                for i in range(len(sym_list)):
                    holder = sym_list[i].split(":")
                    for i in range(len(holder)):
                        sym_hold.append(holder[i])
                sym_list = sym_hold.copy()
                for i in range(len(sym_list)):
                    sym_list[i] = sym_list[i].strip()
                    if sym_list[i] == "hypertension for more than 20 years":
                        sym_list[i] = "hypertension"
                    elif sym_list[i] == "chronic obstructive pulmonary disease":
                        sym_list[i] = "COPD"
                    elif sym_list[i] == "diabetes for more than 20 years":
                        sym_list[i] = "diabetes"
                    elif sym_list[i] == "coronary stenting":
                        sym_list[i] = "coronary heart disease"
                    elif sym_list[i] == "coronary artery stenting":
                        sym_list[i] = "coronary heart disease"
                    elif sym_list[i] == "frequent ventricular premature beat (FVPB)":
                        sym_list[i] = "FVPB"
                    elif sym_list[i] == "Parkinson's disease for five years":
                        sym_list[i] = "Parkinson's disease"
                    elif sym_list[i] == "stenocardia": #This one is a bit debatable
                        sym_list[i] = "coronary heart disease"
                    elif sym_list[i] == "coronary artery stenting":
                        sym_list[i] = "coronary heart disease"
                    elif sym_list[i] == "coronary artery stenting":
                        sym_list[i] = "coronary heart disease"                
                new_x = ""
                for i in range(len(sym_list)):
                    if i == 0 and len(sym_list) == 1:
                        new_x = str(sym_list[i])
                    elif i == 0:
                        new_x = str(sym_list[i]) + ","
                    elif i != len(sym_list)-1:
                        new_x = new_x + " "+str(sym_list[i]) + ","
                    else:
                        new_x = new_x + " " + str(sym_list[i])
            else:
                new_x = x
            return new_x
        
        def chronic_encode(x):
            encode = [-1]*4
            if type(x) == str:
                encode = [0]*4
                sym_list = x.split(",")
                for i in range(len(sym_list)):
                    sym_list[i] = sym_list[i].strip()
                    if sym_list[i] == 'diabetes':
                        encode[3] = 1
                    if sym_list[i] == 'hypertension':
                        encode[2] = 1
                    if sym_list[i] == 'COPD' or sym_list[i] == 'chronic bronchitis' or sym_list[i] == 'Tuberculosis' or sym_list[i] == 'chronic renal insufficiency' or sym_list[i] == 'coronary heart disease' or sym_list[i] == 'colon cancer surgery' or sym_list[i] == 'lung cancer' or sym_list[i] == 'HIV positive' or sym_list[i] == 'prostate hypertrophy' or sym_list[i] == 'Chronic pulmonary condition' or sym_list[i] == 'Pre-renal azotemia' or sym_list[i] == 'asthma' or sym_list[i] == 'valvular heart disease' or sym_list[i] == 'ischemic heart disease' or sym_list[i] == 'benign prostatic hyperplasia' or sym_list[i] == 'dislipidemia' or sym_list[i] == 'atherosclerosis' or sym_list[i] == 'cardiac disease' or sym_list[i] == 'prostate cancer':
                        encode[1] = 1
                    if sym_list[i] == 'COPD' or sym_list[i] == 'chronic bronchitis' or sym_list[i] == 'Tuberculosis' or sym_list[i] == 'chronic renal insufficiency' or sym_list[i] == 'coronary heart disease' or sym_list[i] == 'colon cancer surgery' or sym_list[i] == 'lung cancer' or sym_list[i] == 'HIV positive' or sym_list[i] == 'prostate hypertrophy' or sym_list[i] == 'Chronic pulmonary condition' or sym_list[i] == 'Pre-renal azotemia' or sym_list[i] == 'asthma' or sym_list[i] == 'valvular heart disease' or sym_list[i] == 'ischemic heart disease' or sym_list[i] == 'benign prostatic hyperplasia' or sym_list[i] == 'dislipidemia' or sym_list[i] == 'atherosclerosis' or sym_list[i] == 'cardiac disease' or sym_list[i] == 'prostate cancer' or sym_list[i] == 'diabetes' or sym_list[i] == 'hypertension':
                        empty = 0
                    else:
                        encode[0] = 1
            return encode
        
        #Encode Chronic Disease
        data_mc.loc[data_mc['chronic_disease'] == 'Iran; Kuala Lumpur, Federal Territory of Kuala Lumpur, Malaysia','chronic_disease'] = np.nan
        data_mc.loc[:,"chronic_disease"] = data_mc.loc[:,"chronic_disease"].apply(chronic_disease_standardize) 
        data_mc.loc[:,"chronic_disease"] = data_mc.loc[:,"chronic_disease"].apply(chronic_encode) 
        holder = data_mc.chronic_disease.apply(pd.Series)
        holder.columns = ["Other Underlying","Severe Underlying","hypertension","diabetes"]
        data_mc = pd.concat([data_mc, holder], axis=1, join="inner")
        return data_mc
    
    def __outcome_standardize(self, data_mc):
        """
        Converts the categorical data into a numerical representation
        of a single binary category. See report for details. 0 if discharged,
        1 if died.

        Parameters
        ----------
        data_mc : Pandas Dataframe
            Dataset with categorical outcomes.
        Returns
        -------
        data_mc : Pandas Dataframe
            Dataset with numerically encoded outcomes.

        """     
        def outcome_standardize(x):
            if type(x) == str:
                x.strip()
                sym_list = x.split(",")
                for i in range(len(sym_list)):
                    sym_list[i] = sym_list[i].strip()
                    if sym_list[i] == "dead" or sym_list[i] == "Death" or sym_list[i] == "died" or sym_list[i] == "Deceased" or sym_list[i] == "Dead" or sym_list[i] == "Died":
                        sym_list[i] = "death"
                    elif sym_list[i] == "recovered" or sym_list[i] == "recovering at home 03.03.2020" or sym_list[i] == "discharge" or sym_list[i] == "Discharged" or sym_list[i] == "Discharged from hospital" or sym_list[i] == "Recovered" or sym_list[i] == "released from quarantine" or sym_list[i] == "Alive":
                        sym_list[i] = "discharged"
                new_x = ""
                for i in range(len(sym_list)):
                    if i == 0 and len(sym_list) == 1:
                        new_x = str(sym_list[i])
                    elif i == 0:
                        new_x = str(sym_list[i]) + ","
                    elif i != len(sym_list)-1:
                        new_x = new_x + " " + str(sym_list[i]) + ","
                    else:
                        new_x = new_x + " " + str(sym_list[i])
            else:
                new_x = x
            return new_x
        
        def outcome_encode(x):
            if type(x) == str:
                sym_list = x.split(",")
                for i in range(len(sym_list)):
                    sym_list[i] = sym_list[i].strip()
                    if sym_list[i] == "death":
                        out = 1
                    elif sym_list[i] == "discharged":
                        out = 0
                    else:
                        out = -1
                        raise ValueError("Invalid Outcome Entry")
            else:
                out = -1
            return out
        
        holder = ['https://www.mspbs.gov.py/covid-19.php','not hospitalized','Hospitalized',
                  'Under treatment','Receiving Treatment','Migrated','Migrated_Other',
                  'Symptoms only improved with cough. Currently hospitalized for follow-up.',
                  'Critical condition','critical condition, intubated as of 14.02.2020',
                  'treated in an intensive care unit (14.02.2020)','severe','severe illness',
                  'unstable','critical condition', 'stable', 'Stable', 'stable condition']
        
        for i in range(len(holder)):
            data_mc.loc[data_mc['outcome'] == holder[i],'outcome'] = np.nan
        
        #Encode Outcome    
        data_mc.loc[:,"outcome"] = data_mc.loc[:,"outcome"].apply(outcome_standardize) 
        hold_data = data_mc.assign(bool_outcome = pd.isna(data_mc.loc[:,"outcome"]))
        holder_data = hold_data.query('bool_outcome == False')
        data_mc = holder_data.loc[:,["age", "sex_encoded", "latitude", "longitude",
                              "date_onset_symptoms_days", "date_admission_hospital_days", 
                              "outcome", "symptoms", "diabetes", "hypertension", 
                              "Severe Underlying", "Other Underlying"]]
        
        data_mc.loc[:,"outcome"] = data_mc.loc[:,"outcome"].apply(outcome_encode)
        return data_mc
                
    def __age_cleaner(self, data_mc):
        """
        Fixes range categories in age column. Selects the central value in a 
        range and rounds down for age

        Parameters
        ----------
        data_mc : Pandas Dataframe
            Dataset with ranged ages.
        Returns
        -------
        data_mc : Pandas Dataframe
            Dataset with numerical integer ages.


        """
        #Clean Rest of the Data
        data_mc.loc[data_mc['age'] == '50-59','age'] = "55"
        data_mc.loc[data_mc['age'] == '38-68','age'] = "53"
        data_mc.loc[data_mc['age'] == '80-89','age'] = "85"
        data_mc.loc[data_mc['age'] == '40-49','age'] = "45"
        data_mc.loc[data_mc['age'] == '60-69','age'] = "65"
        data_mc.loc[data_mc['age'] == '20-29','age'] = "25"
        data_mc.loc[data_mc['age'] == '22-80','age'] = "51"
        data_mc.loc[data_mc['age'] == '19-77','age'] = "48"
        data_mc.loc[data_mc['age'] == '70-79','age'] = "75"
        data_mc.loc[data_mc['age'] == '21-72','age'] = "47"
        data_mc.loc[data_mc['age'] == '90-99','age'] = "95"
        data_mc.loc[data_mc['age'] == '15-88','age'] = "52"
        data_mc.loc[data_mc['age'] == '20-57','age'] = "39"
        data_mc.loc[data_mc['age'] == '80-','age'] = "85"
        data_mc.loc[data_mc['age'] == '0.75','age'] = "0"
        data_mc.loc[data_mc['age'] == '0.5','age'] = "0"
        data_mc.loc[data_mc['age'] == '0.25','age'] = "0"
        return data_mc
    
    def __normalize(self, data_mc):
        """
        Normalizes data. Fills all empty entries with -1, before normalizing
        all values in range between 0 and 1.

        Parameters
        ----------
        data_mc : Pandas Dataframe
            Dataset with unnormalized values.
        Returns
        -------
        data_mc : Pandas Dataframe
            Dataset with normalized numerical values.

        """
        columns = ["age", "sex_encoded", "latitude", "longitude",
                      "date_onset_symptoms_days", "date_admission_hospital_days", 
                      "outcome", "symptoms", 
                      "diabetes", "hypertension", "Severe Underlying", 
                      "Other Underlying"]
        #Fill empty entries, and convert age to numerical integer
        data_mc.fillna(value = -1, axis = 1, inplace = True)
        data_mc['age'] = data_mc['age'].astype(int)
        #Normalize Across Datasets
        def normalize(column):
            all_val = data_mc[column]
            try:
                min_val = np.min(all_val)
                max_val = np.max(all_val)
            except:
                raise TypeError("All columns must be numerical to normalize")
            def __norm(x):
                return(x - min_val)/(max_val-min_val)
            data_mc.loc[:,column] = data_mc.loc[:,column].apply(__norm)
            return
        
        for i in range(len(columns)):
            normalize(columns[i])
        return data_mc
    
    def clean_and_save(self):
        data_mc = self.__remove_no_outcome(self.data_pc)
        data_mc = self.__convert_dates(data_mc, "date_death_or_discharge")
        data_mc = self.__convert_dates(data_mc, "date_onset_symptoms")
        data_mc = self.__convert_dates(data_mc, "date_admission_hospital")
        data_mc = self.__sex_standardize(data_mc)
        data_mc = self.__symptom_standardize(data_mc)
        data_mc = self.__chronic_disease_standardize(data_mc)
        data_mc = self.__outcome_standardize(data_mc)
        data_mc = self.__age_cleaner(data_mc)
        data_mc = self.__normalize(data_mc)
        data_mc.to_csv(self.save_path, index=False)
        return data_mc
