import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pickle
import streamlit.components.v1 as components


# Load the InceptionV3 model
#inception_model = InceptionV3(weights='imagenet')

# Everything u write here goes to "tab name"
st.set_page_config(
    page_title = "About Us",
    page_icon = "https://www.freepnglogos.com/uploads/breast-cancer-ribbon-png/breast-cancer-ribbon-symmetry-electronics-supporting-breast-cancer-awareness-8.png")

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css(r"C:\Users\Najeeb\Desktop\Sem 4\Breast Cancer\style\style.css")

def ChangeWidgetFontSize(wgt_txt, wch_font_size = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('p'), i;
                for (i = 0; i < elements.length; ++i) 
                    { if (elements[i].textContent.includes(|wgt_txt|)) 
                        { elements[i].style.fontSize ='""" + wch_font_size + """'; } }</script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)


# Load Animation
animation_symbol = "🎗️"

st.markdown(
    f"""
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    """,
    unsafe_allow_html=True,
)


st.title("Our Objectives!")
st.write("----")
st.write("- The **Objective** of this study was to investigate the effectiveness of supervised machine learning for breast cancer prediction using ultrasound images.")
st.write("- The **Aim** was to develop a model that could accurately classify ultrasound images as malignant, benign, or normal, potentially aiding in early detection and treatment.") 
st.write("- To **Explore the potential** of Supervised Machine Learning in improving the classification accuracy of breast cancer prediction.")

st.title("About Our Dataset!")
st.write("----")
st.write("- Our **Breast Ultrasound Images Dataset** from Sciencedirect.com contains 780 ultrasound images of women aged 25 to 75 years old.") 
st.write("- It was collected in 2018 from 600 female patients and includes original and annotated images. The images are in PNG format and have an average size of 500 x 500 pixels. The dataset has three categories: benign, malignant and normal. ")

st.sidebar.markdown('''
Created with 💜 by [Shio](https://www.instagram.com/thiscloudbook/).
''')