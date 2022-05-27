import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
import seaborn as sns
import os

def app():
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        # df_analysis = pd.read_csv('data/2015.csv')
        df = pd.read_csv('data/main_data.csv')
        
        st.markdown("## Prediction using XGBoost Model")
        st.subheader("Child's Obesity Prediction Form")
        
        """ Cleaning """
        # BMI Class
        def bmi_class(percentile):
            if  percentile == '95+':
                return 'Obese'
            elif percentile == '85 to <95':
                return 'Overweight'
            elif percentile == '5 to <85':
                return 'Normal weight'
            elif percentile == '<5':
                return 'Underweight'
            else:
                return 'null'

        # create a new column based on condition
        df['bmi_class'] = df['BMI_Percentile'].apply(bmi_class)

        # Is Overweight?
        def is_overweight(bmi_class):
            if bmi_class == 'Overweight':
                return 'Yes'
            else:
                return 'No'

        # create a new column based on condition
        df['Patient_is_Overwiegh'] = df['bmi_class'].apply(is_overweight)

        ## Is Obese?
        def is_obese(bmi_class):
            if bmi_class == 'Obese':
                return 'Yes'
            else:
                return 'No'

        # create a new column based on condition
        df['Obesity_identified'] = df['bmi_class'].apply(is_obese)

        ## Over/Below Age 10?
        def age_ten(age):
            if  age == 'over 10 years-adolescents':
                return 1
            else:
                return 0

        # Adjusts value of Age10yrs column
        df['Age10yrs'] = df['Age10yrs'].apply(age_ten)

        ## Imputing the height column
        from sklearn.impute import SimpleImputer, KNNImputer

        imputer = SimpleImputer(strategy='mean')
        imputer = imputer.fit(df[['height']])
        df['height'] = imputer.transform(df[['height']])

        ## Calculating BMI

        from numpy import NaN
        import math

        df['height'] = df['height'] / 100.0
        df['BMIc'] = df['weight'] / (df['height'] * df['height'])

        df = df[df['BMI_Percentile'].notna()] 
        # Create df_pred
        df_pred = df
        df_pred = df_pred.drop('BMC_perecent_481', axis = 1, inplace  = False)
        df_pred = df_pred.drop('Phy_Action', axis = 1, inplace = False)
        df_pred = df_pred.drop('Physicians_Actions_other', axis = 1, inplace  = False)

        # @st.cache(allow_output_mutation=True)
        # @st.cache(suppress_st_warning=True)
        # @st.cache(persist = True)
        
        """ Encoding """

        # creating instance of labelencoder
        labelencoder = LabelEncoder()

        #Assigning numerical values and storing in another column
        df_pred['Patient_seen_at'] = labelencoder.fit_transform(df_pred['Patient_seen_at'])
        df_pred['Gender'] = labelencoder.fit_transform(df_pred['Gender'])
        df_pred['BMI_Percentile'] = labelencoder.fit_transform(df_pred['BMI_Percentile'])
        df_pred['Patient_is_Overwiegh'] = labelencoder.fit_transform(df_pred['Patient_is_Overwiegh'])
        df_pred['Obesity_identified'] = labelencoder.fit_transform(df_pred['Obesity_identified'])
        df_pred['Age10yrs'] = labelencoder.fit_transform(df_pred['Age10yrs'])
        df_pred['Diagnosis'] = labelencoder.fit_transform(df_pred['Diagnosis'])
        df_pred['Diagnosis_other'] = labelencoder.fit_transform(df_pred['Diagnosis_other'])
        #df_pred['bmi_class'] = labelencoder.fit_transform(df_pred['bmi_class'])

        def enc_bmi_class(bmi):
            if  bmi == 'Obese':
                return 0
            elif bmi == 'Overweight':
                return 1
            elif bmi == 'Normal weight':
                return 2
            elif bmi == 'Underweight':
                return 3
            else:
                return 'null'
        df_pred['bmi_class'] = df_pred['bmi_class'].apply(enc_bmi_class)
        ## Old data viz part was here

        """ Balancing """
        
        # Balancing the dataset
        df_balanced = df_pred
        df_balanced.drop(df_pred.columns[6], axis = 1, inplace=True)


        features = []
        for feature in df_balanced.columns:
            if feature != 'bmi_class':
                features.append(feature)
        X = df_balanced[features]
        y = df_balanced['bmi_class']

        # import library
        from imblearn.over_sampling import RandomOverSampler
        from collections import Counter

        ros = RandomOverSampler(random_state=42)

        # fit predictor and target variable
        x_ros, y_ros = ros.fit_resample(X, y)

        df_balanced = x_ros.join(y_ros)


        #""" Splitting into Training and Tetsing """
        
        # Training and Testing
        #st.markdown("#### Train Test Splitting")
        size = 0.8
        array = df_balanced.values
        x = array[:,0:10]
        y = array[:,11]

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=size,
                                                            random_state=42)
        
        def pred_bmi_class(num):
            if  num == 0:
                return 'Obese'
            elif num == 1:
                return 'Overweight'
            elif num == 2:
                return 'Normal weight'
            elif num == 3:
                return 'Underweight'
            else:
                return 'null'
        
        # User Prediction Form
        
        # Clinic
        select_clinic = st.radio(
            'Clinic/Hospital Type?',
            ['Pediatrics Clinic', 'Primary Care Center', 'Other']
        )
        def clinic_type(clinic):
            if  clinic == 'Pediatrics Clinic':
                return 0
            elif clinic == 'Primary Care Center':
                return 1
            else:
                return 2
        select_clinic = clinic_type(select_clinic)

        # Gender
        select_gender = st.radio(
            'Child\'s Gender?',
            ['Female', 'Male'],
        )
        def child_gender(gender):
            if  gender == 'Female':
                return 0
            else:
                return 1
        select_gender = child_gender(select_gender)
        
        # Age
        select_age = st.number_input(
            'Child\'s Age?',
            min_value=1,
            max_value=99
        )
        
        # Height
        select_height = st.number_input(
            'Child\'s Height? [in meters(m)]',
            min_value=0.01,
            step=0.01,
            help='The value must be in meters (m)'
        )
        
        # Weight
        select_weight = st.number_input(
            'Child\'s Weight? [in kilograms(kg)]',
            min_value=0.01,
            step=0.01,
            help='The value must be in kilograms (kg)'
        )
        
        # BMIc
        calc_bmic = select_weight / (select_height * select_height)
        
        # Overweight?
        if calc_bmic >= 25.0 and calc_bmic < 30.0:
            is_overweight=1
        else:
            is_overweight=0
        
        # Obese?
        if calc_bmic >= 30.0:
            is_obese=1
        else:
            is_obese=0
        
        # Older than 10 years?
        if select_age >= 10:
            is_ten=1
        else:
            is_ten=0
        
        # Diagnosis
        select_diagnosis = st.selectbox(
            'Select Child\'s Visit Reason or Diagnosis:',
            ['Other', 'Abdominal Pain', 'Annual Check', 'Bronchial Asthma', 'Constipation', 'Developmental',
             'Diabetes', 'Down\'s Syndrome', 'Enuresis', 'Gastroenteritis', 'Headache', 'Hypothyroidism',
             'Medication Reconciliation', 'Otitis Media', 'Seizure', 'Upper Respiratory', 'Urinary Tract Infection',
             'Vaccination', 'Vitamin D Deficiency']
        )
        def child_diagnosis(diagnosis):
            if  diagnosis == 'Abdominal Pain':
                return 0
            elif diagnosis == 'Annual Check':
                return 1
            elif diagnosis == 'Bronchial Asthma':
                return 2
            elif diagnosis == 'Constipation':
                return 3
            elif diagnosis == 'Developmental':
                return 4
            elif diagnosis == 'Diabetes':
                return 5
            elif diagnosis == 'Down\'s Syndrome':
                return 6
            elif diagnosis == 'Enuresis':
                return 7
            elif diagnosis == 'Gastroenteritis':
                return 8
            elif diagnosis == 'Headache':
                return 9
            elif diagnosis == 'Hypothyroidism':
                return 10
            elif diagnosis == 'Medication Reconciliation':
                return 11
            elif diagnosis == 'Other':
                return 12
            elif diagnosis == 'Otitis Media':
                return 13
            elif diagnosis == 'Seizure':
                return 14
            elif diagnosis == 'Upper Respiratory':
                return 15
            elif diagnosis == 'Urinary Tract Infection':
                return 16
            elif diagnosis == 'Vaccination':
                return 17
            else:
                return 18
        select_diagnosis = child_diagnosis(select_diagnosis)
        
        # x_user
        x_user = [select_clinic, select_gender, select_age, select_height, select_weight, 
                  calc_bmic, is_overweight, is_obese, is_ten, select_diagnosis] 
        x_user = np.array(x_user)
        x_user = np.reshape(x_user, (1, 10))
        
        # Make predictions using XGBoost Model
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(x_train, y_train)
        if st.button('Predict'):
            prediction = model.predict(x_user)
            prediction = pred_bmi_class(prediction)
            st.success(f'Prediction: The child will become {prediction}.')


        
        