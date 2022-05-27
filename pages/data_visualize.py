import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

def app():
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        df_analysis = pd.read_csv('data/main_data.csv')
        st.markdown("#### Your dataset:")
        st.write(df_analysis)
        
    global numeric_columns
    global non_numeric_columns
    try:
        numeric_columns = list(df_analysis.select_dtypes(['float', 'int']).columns)
        non_numeric_columns = list(df_analysis.select_dtypes(['object']).columns)
        non_numeric_columns.append(None)
        print(non_numeric_columns)
    except Exception as e:
        print(e)
        st.write("Please upload file to the application.")
    
    # Add a sidebar
    st.sidebar.subheader("Visualization Settings")
    # add a select widget to the side bar
    chart_select = st.sidebar.selectbox("Chart Type", ("Scatterplots", "Histogram", "Boxplot", "Pie Chart", "Correlation Map"))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if chart_select == 'Scatterplots':
        st.markdown("#### Scatterplot Settings")
        try:
            x_values = st.selectbox('X axis', options=numeric_columns)
            y_values = st.selectbox('Y axis', options=numeric_columns)
            color_value = st.selectbox("Color", options=non_numeric_columns)
            plot = px.scatter(data_frame=df_analysis, x=x_values, y=y_values, color=color_value)
            # display the chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Histogram':
        st.markdown("#### Histogram Settings")
        try:
            x = st.selectbox('Feature', options=numeric_columns)
            bin_size = st.slider("Number of Bins", min_value=10,
                                         max_value=100, value=40)
            color_value = st.selectbox("Color", options=non_numeric_columns)
            plot = px.histogram(x=x, data_frame=df_analysis, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Boxplot':        
        st.markdown("#### Boxplot Settings")
        try:
            y = st.selectbox("Y axis", options=numeric_columns)
            x = st.selectbox("X axis", options=non_numeric_columns)
            color_value = st.selectbox("Color", options=non_numeric_columns)
            plot = px.box(data_frame=df_analysis, y=y, x=x, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)
            
    if chart_select == 'Pie Chart':
        st.markdown("#### Pie Chart Settings")
        try:
            category = st.selectbox("Select Category ", options=non_numeric_columns)
            sizes = (df_analysis[category].value_counts()/df_analysis[category].count())
            labels = sizes.keys()
            maxIndex = np.argmax(np.array(sizes))
            explode = [0]*len(labels)
            explode[int(maxIndex)] = 0.1
            explode = tuple(explode)
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes,explode = explode, labels=labels, autopct='%1.1f%%',shadow=False, startangle=0)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax1.set_title('Distribution for Categorical Column - ' + (str)(category))
            st.pyplot(fig1)
        except Exception as e:
            print(e)
        
    if chart_select == 'Correlation Map':
        try:
            st.markdown("#### Correlation Map")
            corr = df_analysis.corr(method='pearson')
            fig2, ax2 = plt.subplots()
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            # Colors
            cmap = sns.diverging_palette(240, 10, as_cmap=True)
            sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0,ax=ax2)
            ax2.set_title("Correlation Matrix")
            st.pyplot(fig2)
        except Exception as e:
            print(e)