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



st.set_page_config(
    page_title = "Contact Us",
    page_icon = "https://www.freepnglogos.com/uploads/breast-cancer-ribbon-png/breast-cancer-ribbon-symmetry-electronics-supporting-breast-cancer-awareness-8.png")


st.title("Contact Us!")
st.write("---")




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


st.markdown(
    """
    <style>
    .contact-form input[type='text'],
    .contact-form input[type='email'],
    .contact-form textarea {
        display: block;
        width: 100%;
        padding: 8px;
        border: 1px solid #ccc;
        border-radius: 4px;
        box-sizing: border-box;
        margin-bottom: 16px;
        font-size: 16px;
    }
    .contact-form input[type='submit'] {
        background-color: #4CAF50;
        color: white;
        padding: 12px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
    }
    .contact-form input[type='submit']:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define contact form HTML
contact_form = """
    <form class="contact-form" action="https://formsubmit.co/YOUR@MAIL.COM" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <label for="name">Your name:</label>
        <input type="text" id="name" name="name" placeholder="Enter your name" required>
        <label for="email">Your email:</label>
        <input type="email" id="email" name="email" placeholder="Enter your email" required>
        <label for="message">Your message:</label>
        <textarea id="message" name="message" placeholder="Enter your message here" required></textarea>
        <input type="submit" value="Send">
    </form>
"""

# Render contact form with Streamlit
with st.container():
    
    st.header("Get In Touch With Me!")
    st.markdown(contact_form, unsafe_allow_html=True)
    
st.sidebar.markdown('''
Created with 💜 by [Shio](https://www.instagram.com/thiscloudbook/).
''')