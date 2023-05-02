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


# Load the InceptionV3 model
inception_model = InceptionV3(weights='imagenet')

# Everything u write here goes to "tab name"
st.set_page_config(
    page_title = "Breast Cancer Predictor",
    page_icon = "🎗️")

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css(r"C:\Users\Najeeb\Desktop\Sem 4\Breast Cancer\style\style.css")

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

st.title("Breast Cancer Prediction!")

model = tf.keras.models.load_model(r"C:\Users\Najeeb\Desktop\Sem 4\Breast Cancer\tensorflow_model.h5")
#model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

file = st.file_uploader("Please upload an image", type =["jpg", "jpeg", "png"])

def import_and_predict(image_Data, model):
    # Preprocess the input image using InceptionV3
    img = cv2.resize(image_Data, (224, 224))
    x = preprocess_input(np.expand_dims(img, axis=0))
    
    # Make predictions using the loaded TensorFlow model
    predictions = model.predict(x)
    return predictions

def generate_pie_chart(labels, values, colors):
    fig = px.pie(names=labels, values=values)
    fig.update_traces(marker=dict(colors=colors))
    return fig


if file is None:
    st.text("Please upload an image")
else:
    image = Image.open(file).convert('RGB') 
    open_cv_image = np.array(image) 
    # Convert RGB to BGR 
    image = open_cv_image[:, :, ::-1].copy() 
    st.image(image, use_column_width = True)
    predictions = import_and_predict(image, model)
    #st.success(predictions)

    if predictions[0][0] > predictions[0][1]:
        st.success("The image is predicted to be:    'Benign'  with {:.2f}% probability.".format(predictions[0][0]*100))
    elif predictions[0][1] > predictions[0][2]:
        st.success("The image is predicted to be:   'Malignant'  with {:.2f}% probability.".format(predictions[0][1]*100))
    else:
        st.success("The image is predicted to be:    'Normal'  with {:.2f}% probability.".format(predictions[0][2]*100))

    labels = ['Benign', 'Malignant', 'Normal']
    values = predictions[0]
    colors = ['#6eaeee', '#ff3166', '#d39cf1']

    st.plotly_chart(generate_pie_chart(labels, values, colors))