import streamlit as st

# Custom imports 
from multipage import MultiPage
from pages import home, data_upload, data_visualize, machine_learning, evaluation, prediction # import your pages here
from PIL import  Image
import numpy as np

# Create an instance of the app 
app = MultiPage()

# Title of the main page
display = Image.open('logo.jpg')
display = np.array(display)

col1, col2 = st.columns(2)
col1.image(display, width = 350)
col2.title("Childhood Obesity Prediction")

# Add all your applications (pages) here
app.add_page("Home", home.app)
app.add_page("Upload Data", data_upload.app)
app.add_page("Data Visualization", data_visualize.app)
app.add_page("Machine Learning", machine_learning.app)
app.add_page("Model Evaluation", evaluation.app)
app.add_page("Prediction", prediction.app)

# The main app
app.run()