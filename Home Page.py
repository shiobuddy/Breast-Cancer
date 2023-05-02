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
    page_title = "Breast Cancer Predicton",
    page_icon = "https://www.freepnglogos.com/uploads/breast-cancer-ribbon-png/breast-cancer-ribbon-symmetry-electronics-supporting-breast-cancer-awareness-8.png")

st.markdown("""
    <style>
    .stApp {
        margin-left: 0px;
    }
    </style>
    """, unsafe_allow_html=True)    

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
    """,
    unsafe_allow_html=True,
)

st.title("Welcome, Buddy!")

st.header('***Overview***')

st.write('''
**Breast cancer** is cancer that forms in the cells of the breasts.

**Breast cancer** can occur in both men and women, but it's far more common in women.

Substantial support for **breast cancer awareness and research funding** has helped create advances in the diagnosis and treatment of breast cancer.''')



image_url = "https://www.health.pa.gov/topics/programs/PublishingImages/1%20in%208_web.png"
st.image(image_url, use_column_width=True)


st.write("---")


st.header('***Symptoms***')
st.subheader('Signs and symptoms of breast cancer may include:')


st.write('''Signs and symptoms of breast cancer may include:

- A breast lump or thickening that feels different from the surrounding tissue
- Change in the size, shape or appearance of a breast
- Changes to the skin over the breast, such as dimpling
- A newly inverted nipple
- Peeling, scaling, crusting or flaking of the pigmented area of skin surrounding the nipple (areola) or breast skin
- Redness or pitting of the skin over your breast, like the skin of an orange''')


image_url = "https://www.check4cancer.com/images/Advice-Awareness/Be_Breast_Aware_social_tile.jpg"
st.image(image_url, caption='Symptoms Of Breast Cancer', use_column_width=True)

st.write("---")

st.header('***Risk Factors***')

st.write('''A breast cancer risk factor is anything that makes it more likely you'll get breast cancer. But having one or even several breast cancer risk factors doesn't necessarily mean you'll develop breast cancer. Many women who develop breast cancer have no known risk factors other than simply being women.

Factors that are associated with an increased risk of breast cancer include:

- **Being female.** Women are much more likely than men are to develop breast cancer.\n
- **Increasing age.** Your risk of breast cancer increases as you age.
- **A personal history of breast conditions.** If you've had a breast biopsy that found lobular carcinoma in situ (LCIS) or atypical hyperplasia of the breast, you have an increased risk of breast cancer.
- **A personal history of breast cancer.** If you've had breast cancer in one breast, you have an increased risk of developing cancer in the other breast.
- **A family history of breast cancer.** If your mother, sister or daughter was diagnosed with breast cancer, particularly at a young age, your risk of breast cancer is increased. Still, the majority of people diagnosed with breast cancer have no family history of the disease.
- **Inherited genes that increase cancer risk.** Certain gene mutations that increase the risk of breast cancer can be passed from parents to children. The most well-known gene mutations are referred to as BRCA1 and BRCA2. These genes can greatly increase your risk of breast cancer and other cancers, but they don't make cancer inevitable.
- **Radiation exposure.** If you received radiation treatments to your chest as a child or young adult, your risk of breast cancer is increased.
- **Obesity.** Being obese increases your risk of breast cancer.
- **Beginning your period at a younger age.** Beginning your period before age 12 increases your risk of breast cancer.
- **Beginning menopause at an older age.** If you began menopause at an older age, you're more likely to develop breast cancer.
- **Having your first child at an older age.** Women who give birth to their first child after age 30 may have an increased risk of breast cancer.
- **Having never been pregnant.** Women who have never been pregnant have a greater risk of breast cancer than do women who have had one or more pregnancies.
- **Postmenopausal hormone therapy.** Women who take hormone therapy medications that combine estrogen and progesterone to treat the signs and symptoms of menopause have an increased risk of breast cancer. The risk of breast cancer decreases when women stop taking these medications.
- **Drinking alcohol.** Drinking alcohol increases the risk of breast cancer.''')


image_url = "https://cdn.cancercenter.com/-/media/ctca/images/feature-block-images/medical-illustrations/breast-cancer-risk-factors-dtm.png"
st.image(image_url, caption='Risk Factors Of Breast Cancer', use_column_width=True)


st.write("---")


st.header('***Breast cancer risk reduction for women with a high risk***')

st.write('''If your doctor has assessed your family history and determined that you have other factors, such as a precancerous breast condition, that increase your risk of breast cancer, you may discuss options to reduce your risk, such as:

**Preventive medications (chemoprevention).** Estrogen-blocking medications, such as selective estrogen receptor modulators and aromatase inhibitors, reduce the risk of breast cancer in women with a high risk of the disease.

These medications carry a risk of side effects, so doctors reserve these medications for women who have a very high risk of breast cancer. Discuss the benefits and risks with your doctor.

**Preventive surgery.** Women with a very high risk of breast cancer may choose to have their healthy breasts surgically removed (prophylactic mastectomy). They may also choose to have their healthy ovaries removed (prophylactic oophorectomy) to reduce the risk of both breast cancer and ovarian cancer.
''')

st.write("---")

st.sidebar.markdown('''
Created with 💜 by [Shio](https://www.instagram.com/thiscloudbook/).
''')