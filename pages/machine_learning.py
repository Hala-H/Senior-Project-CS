# Libraries

# Initializing the web app created by streamlit
import streamlit as st

# Import necessary libraries
import os
import json
import joblib

# Import visualization and processing libraries
from numpy.core.numeric import True_
from sklearn import metrics
import pandas as pd
from pandas import read_excel
from pandas.plotting import scatter_matrix
import numpy as np
import xgboost as xgb
import seaborn as sns
from scipy import stats
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from matplotlib import figure
import matplotlib.pyplot as plt
import plotly.express as px

# Import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score,GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
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

#Label Encoding on the categorical attributes
from sklearn.preprocessing import LabelEncoder
import imblearn


# App starts here
def app():
    
    """This application helps in running machine learning models without having to write explicit code 
    by the user. It runs some basic models and let's the user select the X and y variables. 
    """    
    # Load the data 
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        df = pd.read_csv('C:\\Users\\hhane\\Desktop\\Spring 2022\\Senior 1\\SP22\\data\\main_data.csv')
    
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

        ## Old data viz part was here
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


        """ Splitting into Training and Tetsing """
        
        # Training and Testing
        st.markdown("#### Train Test Splitting")
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
        st.write("Number of training samples:", x_train.shape[0])
        st.write("Number of testing samples:", x_test.shape[0])
        
        st.markdown("#### Your dataset has been preprocessed, cleaned and is now ready for training!")
        st.write(df_balanced)

        """ Running the Machine Learning Models """
        
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
        
        st.markdown("#### Cross-validation k-fold size")
        size = st.slider("Percentage of value division",
                                min_value= 2, 
                                max_value= 10, 
                                step = 1, 
                                value= 5, 
                                help="This is the value which will be used as the size for the cross-validation folds. Default = 5")
        
        kfold = StratifiedKFold(n_splits=size, random_state=1, shuffle=True)

        # Table to store model and accuracy (f1-score)
        model_f1 = []
        
        if classifier == "Logistic Regression":
            st.write("The model is now being trained and you will get the accuracy.")
            lr = LogisticRegression(solver='liblinear', multi_class='ovr')
            lr.fit(x_train, y_train)
            lr_results = cross_val_score(lr, x_train, y_train, cv = kfold,
                                         scoring='f1_macro') 
            model_f1.append(['Logistic Regression', lr_results.mean()])
            st.write("Logistic Regression's F1-score %.2f" % lr_results.mean())
        
        if classifier == "Linear Discriminant Analysis":
            st.write("The model is now being trained and you will get the accuracy.")
            lda = LinearDiscriminantAnalysis()
            lda.fit(x_train, y_train)
            lda_results = cross_val_score(lda, x_train, y_train, cv = kfold,
                                         scoring='f1_macro') 
            model_f1.append(['Linear Disriminant Analysis', lda_results.mean()])
            st.write("Linear Disriminant Analysis's F1-score %.2f" % lda_results.mean())
        
        if classifier == "Support Vector Machine":
            st.write("The model is now being trained and you will get the accuracy.")
            svm = SVC(gamma='auto', kernel='linear')
            svm.fit(x_train, y_train)
            svm_results = cross_val_score(svm, x_train, y_train, cv = kfold,
                                         scoring='f1_macro') 
            model_f1.append(['Support Vector Machine', svm_results.mean()])
            st.write("Support Vector Machine's F1-score %.2f" % svm_results.mean())
    
        if classifier == "Random Forest":
            st.write("The model is now being trained and you will get the accuracy.")
            rf = RandomForestClassifier(n_estimators = 1344)
            rf.fit(x_train, y_train)
            rf_results = cross_val_score(rf, x_train, y_train, cv = kfold,
                                         scoring='f1_macro') 
            model_f1.append(['Random Forest', rf_results.mean()])
            st.write("Random Forest's F1-score %.3f" % rf_results.mean())
            
        if classifier == "K-Nearest Neighbors":
            st.write("The model is now being trained and you will get the accuracy.")
            knn = KNeighborsClassifier()
            knn.fit(x_train, y_train)
            knn_results = cross_val_score(knn, x_train, y_train, cv = kfold,
                                         scoring='f1_macro') 
            model_f1.append(['K-Nearest Neighbors', knn_results.mean()])
            st.write("K-Nearest Neighbors's F1-score %.2f" % knn_results.mean())
        
        if classifier == "Decision Tree":
            st.write("The model is now being trained and you will get the accuracy.")
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            dt_results = cross_val_score(dt, x_train, y_train, cv = kfold,
                                         scoring='f1_macro') 
            model_f1.append(['Decision Tree', dt_results.mean()])
            st.write("Decision Tree's F1-score %.2f" % dt_results.mean())
            
        if classifier == "AdaBoost":
            st.write("The model is now being trained and you will get the accuracy.")
            ada = AdaBoostClassifier(n_estimators=1344, random_state=1)
            ada.fit(x_train, y_train)
            ada_results = cross_val_score(ada, x_train, y_train, cv = kfold,
                                         scoring='f1_macro') 
            model_f1.append(['AdaBoost', ada_results.mean()])
            st.write("AdaBoost's F1-score %.2f" % ada_results.mean())
            
        if classifier == "XGBoost":
            st.write("The model is now being trained and you will get the accuracy.")
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            xgb.fit(x_train, y_train)
            xgb_results = cross_val_score(xgb, x_train, y_train, cv = kfold,
                                         scoring='f1_macro') 
            model_f1.append(['XGBoost', xgb_results.mean()])
            st.write("XGBoost's F1-score %.2f" % xgb_results.mean())
        
        if classifier == "Naive Bayes":
            st.write("The model is now being trained and you will get the accuracy.")
            nb = MultinomialNB()
            nb.fit(x_train, y_train)
            nb_results = cross_val_score(nb, x_train, y_train, cv = kfold,
                                         scoring='f1_macro') 
            model_f1.append(['Naive Bayes', nb_results.mean()])
            st.write("Naive Bayes's F1-score %.2f" % nb_results.mean())
            
        if st.sidebar.button("All algorithms"): 
            st.write("The model is now being trained using all algorithms and you will get the accuracy.")
            # Spot Check Algorithms
            models = []
            models.append(('LR', LogisticRegression(solver='liblinear',
                                                    multi_class='ovr')))
            models.append(('LDA', LinearDiscriminantAnalysis()))
            models.append(('SVM', SVC(gamma='auto')))
            models.append(('RF', RandomForestClassifier(n_estimators = 1344)))
            models.append(('KNN', KNeighborsClassifier()))
            models.append(('CART', DecisionTreeClassifier()))
            models.append(('AdaBoost', AdaBoostClassifier(
                n_estimators=1344, random_state=1)))
            models.append(('XGBoost',  XGBClassifier(use_label_encoder=False,
                                                     eval_metric='mlogloss')))
            models.append(('NB', MultinomialNB()))

            # evaluate each model in turn
            results_avg = []
            results_all = []
            names = []
            for name, model in models:
                cv_results = cross_val_score(model, x_train, y_train, cv = kfold,
                                             scoring='f1_macro')
                results_avg.append(cv_results.mean())
                results_all.append(cv_results)
                names.append(name)
            res_avg = pd.DataFrame(
                {'Model': names,
                 'F1_Score': results_avg
                })
            st.write(res_avg)
            
            # Boxplot comparison
            res_final = pd.DataFrame(
                {'Model': names,
                 'F1_Score': results_all
                })
            data = []
            for i in res_final.itertuples():
                lst = i[2]
                for col2 in lst:
                    data.append([i[1], col2])
            res_final = pd.DataFrame(data=data, columns=res_final.columns)
            plot = px.box(res_final, y="F1_Score", x="Model", title="Algorithm Comparison")
            st.plotly_chart(plot)
            