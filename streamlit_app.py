import streamlit as st
import pickle
import datetime as dt
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
#import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import keras
import spacy

#load models
@st.cache(allow_output_mutation=True)
def spacy_load(model_name):
    nlp = spacy.load(model_name)
    return nlp
nlp = spacy_load('en_core_web_sm')

@st.cache(allow_output_mutation=True)
def load_model(path):
    model = keras.models.load_model(path)
    return model
model = load_model('pickles/dl_1215_5m_100_log_daily.h5')

@st.cache(allow_output_mutation=True)
def load_parameters(path):
    with open(path, 'rb') as f:
        parameters = pickle.load(f)
        return parameters
scaler = load_parameters('pickles/feature_scaler_1215.pkl')
title_tokenizer = load_parameters('pickles/title_tokenizer_1215.pkl')
tag_tokenizer = load_parameters('pickles/tag_tokenizer_1215.pkl')
des_tokenizer = load_parameters('pickles/des_tokenizer_1215.pkl')


#variables
blank_image = np.full((1,224, 224, 3), 1)
today = dt.date.today()
categories = ['Entertainment', 'Gaming', 'Film & Animation', 'Sports', 'News & Politics', 'People & Blogs', 'Howto & Style', 'Music',
              'Autos & Vehicles', 'Education', 'Travel & Events', 'Science & Technology', 'Comedy', 'Nonprofits & Activism', 'Other']
maxlen_title = 22
maxlen_tag = 105
maxlen_des = 765

#sidebar
st.sidebar.markdown('# YouTube Video Information Evaluation System')
st.sidebar.markdown('## Video Information')
st.sidebar.write('1. The duration of your video:')
duration_cols_1 = st.sidebar.columns(2)
duration_cols_2 = st.sidebar.columns(2)
day = duration_cols_1[0].number_input("Days", min_value = 0)
hour = duration_cols_1[1].number_input("Hours", min_value = 0)
minute = duration_cols_2[0].number_input("Minutes", min_value = 0, value = 5)
second = duration_cols_2[1].number_input("Seconds", min_value = 0)
# day = st.sidebar.number_input("Days", min_value = 0)
# hour = st.sidebar.number_input("Hours", min_value = 0)
# minute = st.sidebar.number_input("Minutes", min_value = 0, value = 5)
# second = st.sidebar.number_input("Seconds", min_value = 0)
category = st.sidebar.selectbox('2. The YouTube video category associated with the video', categories)
vidoe_kid = st.sidebar.selectbox('3. Your video contains the current "made for kids" status', ['No', 'Yes'])
hd = st.sidebar.selectbox('4. Your video is HD or SD?', ['HD', 'SD'])
rectangular = st.sidebar.selectbox('5. Your video is rectangular or 360?', ['Rectangular', '360'])

st.sidebar.markdown('## Channel Information')
date = st.sidebar.date_input('1. The date that your channel was created', min_value = dt.date(2005,12,15), max_value = today)
subscriber = st.sidebar.number_input('2. The number of subscribers that your channel has', min_value = 0)
channel_video = st.sidebar.number_input('3. The number of public videos uploaded to your channel', min_value = 0)
country = st.sidebar.selectbox('4. The country with which your channel is associated', ['US', 'IN', 'GB', 'CA', 'AU', 'Other'])
channel_kid = st.sidebar.selectbox('5. Your channel contains the current "made for kids" status', ['No', 'Yes'])

#page
st.markdown("### You can use this system to optimize your thumbnail, title, tag, and description. Please fill out the information on the sidebar first to get better estimation.")
title_input = st.text_input("The title of your video")
tag_input = st.text_input("The tags of your video")
des_input = st.text_area("The description of your video", height = 1)
img = st.file_uploader('Upload your thumbnail')

if img != None:
    st.image(Image.open(img))
    thumbnail = Image.open(img).convert('RGB').resize((224,224))
    thumbnail = np.array([np.array(thumbnail)/255])
else:
    thumbnail = blank_image


#features
duration = second + minute * 60 + hour * 60 * 60 + day * 60 * 60 * 24
vidoe_kid_yes = 1 if vidoe_kid == 'Yes' else 0
hd_yes = 1 if hd == 'HD' else 0
rectangular_yes = 1 if rectangular == 'Rectangular' else 0
categoryId_1 = 1 if category == 'Film & Animation' else 0
categoryId_10 = 1 if category == 'Music' else 0
categoryId_17 = 1 if category == 'Sports' else 0
categoryId_19 = 1 if category == 'Travel & Events' else 0
categoryId_2 = 1 if category == 'Autos & Vehicles' else 0
categoryId_20 = 1 if category == 'Gaming' else 0
categoryId_22 = 1 if category == 'People & Blogs' else 0
categoryId_23 = 1 if category == 'Comedy' else 0
categoryId_24 = 1 if category == 'Entertainment' else 0
categoryId_25 = 1 if category == 'News & Politics' else 0
categoryId_26 = 1 if category == 'Howto & Style' else 0
categoryId_27 = 1 if category == 'Education' else 0
categoryId_28 = 1 if category == 'Science & Technology' else 0
categoryId_29 = 1 if category == 'Nonprofits & Activism' else 0
channel_kid_yes = 1 if channel_kid == 'Yes' else 0
US = 1 if country == 'US' else 0
IN = 1 if country == 'IN' else 0
GB = 1 if country == 'GB' else 0
CA = 1 if country == 'CA' else 0
AU = 1 if country == 'AU' else 0
channel_duration = (today - date).total_seconds()

features = [duration, vidoe_kid_yes, hd_yes, rectangular_yes, categoryId_1, categoryId_10, 
            categoryId_17, categoryId_19, categoryId_2, categoryId_20, categoryId_22, categoryId_23,
            categoryId_24, categoryId_25, categoryId_26, categoryId_27, categoryId_28, categoryId_29, 
            subscriber, channel_video, channel_kid_yes, US, IN, GB, CA, AU, channel_duration]

test_fea = scaler.transform(np.array(features).reshape(1, -1))

#nlp
title_nlp = [w.lemma_.lower() for w in nlp(title_input) if not w.is_stop and not w.is_punct and not w.like_num]
title = [' '.join(title_nlp)]
tag_nlp = [w.lemma_.lower() for w in nlp(tag_input) if not w.is_stop and not w.is_punct and not w.like_num]
tag = [' '.join(tag_nlp)]
des_nlp = [w.lemma_.lower() for w in nlp(des_input) if not w.is_stop and not w.is_punct and not w.like_num]
des = [' '.join(des_nlp)]

test_title = title_tokenizer.texts_to_sequences(title)
test_title = sequence.pad_sequences(test_title, maxlen=maxlen_title)
test_tag = tag_tokenizer.texts_to_sequences(tag)
test_tag = sequence.pad_sequences(test_tag, maxlen=maxlen_tag)
test_des = des_tokenizer.texts_to_sequences(des)
test_des = sequence.pad_sequences(test_des, maxlen=maxlen_des)

#predict
baseline = model.predict([blank_image, test_fea, np.zeros((1,maxlen_title)), np.zeros((1,maxlen_tag)), np.zeros((1,maxlen_des))])[0][0]
prediction = model.predict([thumbnail, test_fea, test_title, test_tag, test_des])[0][0] 

#st.markdown(f'# {10 ** baseline}')
#st.markdown(f'# {10 ** prediction}')
#st.markdown(f'# Improvement: {int((10 ** prediction/ 10 ** baseline - 1) * 100)}%')
st.markdown(f'# Video Information Score: {int((10 ** (prediction - baseline) - 1) * 100)}% \n'
             '* The number above shows the increasing percentage of views compared to blank image, title, tag and description.\n'
             '* You can upload different thumbnails and enter different titles, tags and description to compare the scores of different combinations)')


