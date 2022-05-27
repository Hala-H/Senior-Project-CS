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
        size = st.slider("Percentage of value division",
                                min_value=0.1, 
                                max_value=0.9, 
                                step = 0.1, 
                                value=0.8, 
                                help="This is the value which will be used to divide the data for training and testing. Default = 80%")
        array = df_balanced.values
        x = array[:,0:10]
        y = array[:,11]

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=size,
                                                            random_state=42)
        
        # Choosing the classifier
        st.sidebar.subheader("Choose classifier")
        classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression",
                                                         "Linear Discriminant Analysis",
                                                         "Support Vector Machine",
                                                         "Random Forest",
                                                         "K-Nearest Neighbors",
                                                         "Decision Tree",
                                                         "AdaBoost",
                                                         "XGBoost",
                                                         "Naive Bayes"))
        
        if classifier == "Logistic Regression":
            model = LogisticRegression(solver='liblinear', multi_class='ovr')
        if classifier == "Linear Discriminant Analysis":
            model = LinearDiscriminantAnalysis()
        if classifier == "Support Vector Machine":
            model = SVC(gamma='auto', kernel='linear')
        if classifier == "Random Forest":
            model = RandomForestClassifier(n_estimators = 1344 )
        if classifier == "K-Nearest Neighbors":
            model = KNeighborsClassifier()
        if classifier == "Decision Tree":
            model = DecisionTreeClassifier()
        if classifier == "AdaBoost":
            model = AdaBoostClassifier(n_estimators=1344, random_state=1)
        if classifier == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        if classifier == "Naive Bayes":
            model = MultinomialNB()
        
        # Make predictions
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        # Print predictions
        st.markdown("#### Prediction Results")
        df_x = pd.DataFrame(x_test, columns=['Patient_seen_at', 'Gender', 'Age', 'Height', 'Weight', 'BMIc', 
                                             'Patient_is_Overweight', 'Obesity_identified', 'Age10yrs', 'Diagnosis'])
        df_y = pd.DataFrame(predictions, columns=['Predicted_bmi_class'])
        df_predicted = df_x.join(df_y)
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
        df_predicted['Predicted_bmi_class'] = df_predicted['Predicted_bmi_class'].apply(pred_bmi_class)
        st.write(df_predicted)
        st.write(df_predicted.groupby('Predicted_bmi_class').size())
        
        # Evaluate predictions
        st.markdown("#### Prediction Evaluation")
        
        accuracy = accuracy_score(y_test, predictions)
        st.write("Accuracy: ", accuracy.round(3))
        
        st.write("Confusion Matrix: ") 
        cm = confusion_matrix(y_test, predictions)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix') 
        ax.xaxis.set_ticklabels(['Obese', 'Overweight', 'Normal', 'Under'])
        ax.yaxis.set_ticklabels(['Obese', 'Overweight', 'Normal', 'Under'])
        st.pyplot()
        
        st.write("Classification Report: ")
        st.text(classification_report(y_test, predictions))
        
        