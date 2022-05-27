import collections
from numpy.core.defchararray import lower
import streamlit as st
import numpy as np
import pandas as pd
from PIL import  Image

def app():
    st.markdown("## Home")

    st.subheader('About')

    # introduction
    st.write("""
    This application is the deployment of childhood obesity prediction model.
    You can check on the sidebar the different services provided:
    - Upload Data
    - Data Visualization
    - Machine Learning
    - Model Evaluation
    - Prediction
    The model aims to predict childhood obesity based on specific paramters based on training different machine learning models in the machine learning page. Then, the final prediction model will be built on the best performing model.
    """)

    st.subheader('Model Definition')

    st.write("""
    The structure of the training is based on cross-validation. The user can determine the number of k-folds and it will be trained based on that. Then, the training accuracy will be shown. Next, in the evaluation, the models are tested using the testing dataset and the final results are shown in a confusion matrix and classification report. The final prediction model which the end user will use will be built on the best performing model
    The combinations are regarding to perform Feature Creation and/or Target Transformations in the dataset.
    Models:
    - Logistic Regression
    - Linear Discriminant Analysis
    - Support Vector Machine
    - Random Forest
    - K-Nearest Neighbors
    - Decision Tree
    - AdaBoost
    - XGBoost
    - Naive Bayes

    Our main accuracy metric is F1-Score. To enhance our model definition, we utilized Cross Validation and Random Search for hyperparameter tuning.
    """)