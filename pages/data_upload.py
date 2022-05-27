import collections
from numpy.core.defchararray import lower
import streamlit as st
import numpy as np
import pandas as pd

def app():
    st.markdown("## Upload Data")

    # Upload the dataset and save as csv
    st.markdown("### Upload a csv or excel file for analysis.") 
    st.write("\n")

    # Code to read a single file 
    uploaded_file = st.file_uploader("Choose a file", type = ['csv', 'xlsx'])
    global df
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            df = pd.read_excel(uploaded_file)

    ''' Load the data and save the columns with categories as a dataframe. 
    This section also allows changes in the numerical and categorical columns. '''
    if st.button("Load Data"):
        
        # Raw data 
        st.dataframe(df)
        df.to_csv('data/main_data.csv', index=False)