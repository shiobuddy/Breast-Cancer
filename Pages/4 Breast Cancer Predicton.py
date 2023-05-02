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
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image, ImageOps
from sklearn.decomposition import PCA

scaler = StandardScaler()

# Everything u write here goes to "tab name"
st.set_page_config(
    page_title = "Breast Cancer Predictor",
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


def get_glcm_feature(image):
    # Compute the gray-level co-occurrence matrix (GLCM)
    glcm = greycomatrix(image, [1], [0], levels=256, symmetric=True, normed=True)
    # Compute the contrast, dissimilarity, homogeneity, ASM, energy, and correlation from the GLCM
    contrast = greycoprops(glcm, 'contrast')[0][0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0][0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0][0]
    ASM = greycoprops(glcm, 'ASM')[0][0]
    energy = greycoprops(glcm, 'energy')[0][0]
    correlation = greycoprops(glcm, 'correlation')[0][0]
    return [contrast, dissimilarity, homogeneity, ASM, energy, correlation]

st.sidebar.markdown('''
Created with 💜 by [Shio](https://www.instagram.com/thiscloudbook/).
''')


st.title("Breast Cancer Prediction!")

model = tf.keras.models.load_model(r"C:\Users\Najeeb\Desktop\Sem 4\Breast Cancer\tensorflow_model.h5")

#model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


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

def load_model(model_path):
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    return data



file = st.file_uploader("Please upload an image", type =["jpg", "jpeg", "png"])

if file is None:
    st.text("Please upload an image")
else:

    listTabs = ['Recommended','For Advanced Users']
    tab1, tab2 = st.tabs(listTabs)
    ChangeWidgetFontSize(listTabs[0], '19px')
    ChangeWidgetFontSize(listTabs[1], '19px')

    with tab1:

        button_clicked = st.button("Predict!!")

        # Check if the button has been clicked
        if button_clicked:

            image = Image.open(file).convert('RGB') 
            open_cv_image = np.array(image) 
            # Convert RGB to BGR 
            image = open_cv_image[:, :, ::-1].copy() 
            st.image(image, use_column_width = True)
            predictions = import_and_predict(image, model)
            #st.success(predictions)

            if predictions[0][0] > predictions[0][1]:
                st.subheader("The image is predicted to be: **Cancerous** (Benign)  with {:.2f}% probability.".format(predictions[0][0]*100))
            elif predictions[0][1] > predictions[0][2]:
                st.subheader("The image is predicted to be: **Cancerous** (Malignant)  with {:.2f}% probability.".format(predictions[0][1]*100))
            else:
                st.subheader("The image is predicted to be: **Non-Cancerous** (Normal)  with {:.2f}% probability.".format(predictions[0][2]*100))

            labels = ['Benign', 'Malignant', 'Normal']
            values = predictions[0]
            colors = ['#FEBA4F', '#ff3166', '#00A693']

            st.plotly_chart(generate_pie_chart(labels, values, colors))


            with open(r"C:\Users\Najeeb\Desktop\Sem 4\Breast Cancer Code\Source Code\history_inception.pkl", 'rb') as f:
                history = pickle.load(f)

            with st.expander("Click To Display Accuracy And Losss Of The Model"):

                fig, ax = plt.subplots()
                ax.plot(history['accuracy'], label='train_accuracy')
                ax.plot(history['val_accuracy'], label='val_accuracy')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Accuracy')
                ax.legend()
                st.pyplot(fig)

                fig, ax = plt.subplots()
                ax.plot(history['loss'], label='train_loss')
                ax.plot(history['val_loss'], label='val_loss')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend()
                st.pyplot(fig)


    with tab2:
        # Dictionary to store the model names and their corresponding preprocessing techniques
        models = {
            'SVM (OTSU thresholding)': r'C:\Users\Najeeb\Desktop\Model\SVM_OTSU.pkl',
            'PCA+SVM (OTSU thresholding)': r'C:\Users\Najeeb\Desktop\Model\SVM_PCA_OTSU.pkl',
            'LDA+SVM (OTSU thresholding)': r'C:\Users\Najeeb\Desktop\Model\SVM_LDA_OTSU.pkl',
            'Decision Tree (OTSU thresholding)': r'C:\Users\Najeeb\Desktop\Model\DT_OTSU.pkl',
            'Random Forest (OTSU thresholding)': r'C:\Users\Najeeb\Desktop\Model\RF_OTSU.pkl',
            'Random Forest (GLCM feature extraction and scaling)': r'C:\Users\Najeeb\Desktop\Model\RF_GLCM.pkl',
            'SVM (GLCM feature extraction and scaling)': r'C:\Users\Najeeb\Desktop\Model\SVM_GLCM.pkl',
            'Inception V3': 'inception_v3.joblib'
        }

        model_choice = st.selectbox("Select a model", list(models.keys()))

        # Load the selected model
        model_path = models[model_choice]

        button_clicked = st.button("Predict!")

        # Check if the button has been clicked
        if button_clicked:

            if model_path == r'C:\Users\Najeeb\Desktop\Model\SVM_OTSU.pkl':

                model = load_model(model_path)
                
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image = image.copy()

                
                st.image(image, caption='Uploaded Image', use_column_width=True)


                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                image = thresh

                image = image.flatten()
                
                image = [image]

                prediction = model.predict(image)

                if prediction[0] == 'benign':
                    st.subheader("The image is predicted to be: **Cancerous** (Benign)")
                elif prediction[0] == 'malignant':
                    st.subheader("The image is predicted to be: **Cancerous** (Malignant)")
                else:
                    st.subheader("The image is predicted to be: **Non-Cancerous** (Normal)")


            if model_path == r'C:\Users\Najeeb\Desktop\Model\SVM_PCA_OTSU.pkl':

                model = load_model(model_path)
                
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image = image.copy()

                
                st.image(image, caption='Uploaded Image', use_column_width=True)


                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                image = thresh

                image = image.flatten()
                
                image = [image]

                pca = joblib.load(r"C:\Users\Najeeb\Desktop\Model\SVM_PCA.pkl")


                image = pca.transform(image)

                prediction = model.predict(image)

                if prediction[0] == 'benign':
                    st.subheader("The image is predicted to be: **Cancerous** (Benign)")
                elif prediction[0] == 'malignant':
                    st.subheader("The image is predicted to be: **Cancerous** (Malignant)")
                else:
                    st.subheader("The image is predicted to be: **Non-Cancerous** (Normal)")

            
            if model_path == r'C:\Users\Najeeb\Desktop\Model\SVM_LDA_OTSU.pkl':

                model = load_model(model_path)
                
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image = image.copy()

                
                st.image(image, caption='Uploaded Image', use_column_width=True)


                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                image = thresh
                image = image.flatten()
                image = [image]

                pca = joblib.load(r"C:\Users\Najeeb\Desktop\Model\SVM_PCA.pkl")
                image = pca.transform(image)

                prediction = model.predict(image)

                if prediction[0] == 'benign':
                    st.subheader("The image is predicted to be: **Cancerous** (Benign)")
                elif prediction[0] == 'malignant':
                    st.subheader("The image is predicted to be: **Cancerous** (Malignant)")
                else:
                    st.subheader("The image is predicted to be: **Non-Cancerous** (Normal)")


            if model_path == r'C:\Users\Najeeb\Desktop\Model\DT_OTSU.pkl':

                model = load_model(model_path)
                
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image = image.copy()

                
                st.image(image, caption='Uploaded Image', use_column_width=True)


                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                image = thresh

                image = image.flatten()
                
                image = [image]

                prediction = model.predict(image)

                if prediction[0] == 'benign':
                    st.subheader("The image is predicted to be: **Cancerous** (Benign)")
                elif prediction[0] == 'malignant':
                    st.subheader("The image is predicted to be: **Cancerous** (Malignant)")
                else:
                    st.subheader("The image is predicted to be: **Non-Cancerous** (Normal)")


            if model_path == r'C:\Users\Najeeb\Desktop\Model\RF_OTSU.pkl':

                model = load_model(model_path)
                
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image = image.copy()

                
                st.image(image, caption='Uploaded Image', use_column_width=True)


                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                image = thresh

                image = image.flatten()
                
                image = [image]

                prediction = model.predict(image)
                if prediction[0] == 'benign':
                    st.subheader("The image is predicted to be: **Cancerous** (Benign)")
                elif prediction[0] == 'malignant':
                    st.subheader("The image is predicted to be: **Cancerous** (Malignant)")
                else:
                    st.subheader("The image is predicted to be: **Non-Cancerous** (Normal)")


            if model_path == r"C:\Users\Najeeb\Desktop\Model\SVM_GLCM.pkl":

                model = load_model(model_path)
                
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                
                st.image(image, caption='Uploaded Image', use_column_width=True)

                image = image.copy()
                
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                #image = np.reshape(image, (image.shape[0], image.shape[1], 3))
                #st.success(image.shape())

                _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                image = thresh
                
                # Extract features from the image
                features = [get_glcm_feature(image)]
                
                
                # Preprocess the data by scaling the features
                scaler = joblib.load(r"C:\Users\Najeeb\Desktop\Model\SVM_GLCM_scaler.pkl")
                features_scaled = scaler.transform(features)
                
                
                # Make a prediction using the trained Random Forest model
                prediction = model.predict(features_scaled)
                
                if prediction == np.array([1]):
                    st.subheader("The image is predicted to be: **Cancerous**")
                else:
                    st.subheader("The image is predicted to be: **Non-Cancerous**")



            if model_path == r"C:\Users\Najeeb\Desktop\Model\RF_GLCM.pkl":

                model = load_model(model_path)
                
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                
                st.image(image, caption='Uploaded Image', use_column_width=True)

                image = image.copy()
                
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                #image = np.reshape(image, (image.shape[0], image.shape[1], 3))
                #st.success(image.shape())

                _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                image = thresh
                
                # Extract features from the image
                features = [get_glcm_feature(image)]
                
                # Preprocess the data by scaling the features
                scaler = joblib.load(r'C:\Users\Najeeb\Desktop\Model\RF_GLCM_scaler.pkl')
                features_scaled = scaler.transform(features)
                
                # Make a prediction using the trained Random Forest model
                prediction = model.predict(features_scaled)
                
                if prediction == np.array([1]):
                    st.subheader("The image is predicted to be: **Cancerous**")
                else:
                    st.subheader("The image is predicted to be: **Non-Cancerous**")


            if model_path == "inception_v3.joblib":

                image = Image.open(file).convert('RGB') 
                open_cv_image = np.array(image) 
                # Convert RGB to BGR 
                image = open_cv_image[:, :, ::-1].copy() 
                st.image(image, use_column_width = True)
                predictions = import_and_predict(image, model)
                #st.success(predictions)

                if predictions[0][0] > predictions[0][1]:
                    st.subheader("The image is predicted to be: **Cancerous** (Benign)  with {:.2f}% probability.".format(predictions[0][0]*100))
                elif predictions[0][1] > predictions[0][2]:
                    st.subheader("The image is predicted to be: **Cancerous** (Malignant)  with {:.2f}% probability.".format(predictions[0][1]*100))
                else:
                    st.subheader("The image is predicted to be: **Non-Cancerous** (Normal)  with {:.2f}% probability.".format(predictions[0][2]*100))

                labels = ['Benign', 'Malignant', 'Normal']
                values = predictions[0]
                colors = ['#FEBA4F', '#ff3166', '#00A693']

                st.plotly_chart(generate_pie_chart(labels, values, colors))


                with open(r"C:\Users\Najeeb\Desktop\Sem 4\Breast Cancer Code\Source Code\history_inception.pkl", 'rb') as f:
                    history = pickle.load(f)

                with st.expander("Click To Display Accuracy And Losss Of The Model"):

                    fig, ax = plt.subplots()
                    ax.plot(history['accuracy'], label='train_accuracy')
                    ax.plot(history['val_accuracy'], label='val_accuracy')
                    ax.set_xlabel('Epochs')
                    ax.set_ylabel('Accuracy')
                    ax.legend()
                    st.pyplot(fig)

                    fig, ax = plt.subplots()
                    ax.plot(history['loss'], label='train_loss')
                    ax.plot(history['val_loss'], label='val_loss')
                    ax.set_xlabel('Epochs')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    st.pyplot(fig)