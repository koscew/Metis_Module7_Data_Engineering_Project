import streamlit as st
import pickle
import datetime as dt
import numpy as np
import pandas as pd
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

#page_A
st.markdown("## You can use this system to optimize your thumbnail, title, tag, and description. \n"
            "1. You can fill out the information on the sidebar first to get better estimation. \n"
            "2. Enter the title, tags and description and upload the thumbnail of your video into ***Example A*** \n"
            "3. You can choose to either enter the title, tags and description and upload the thumbnail of your video into ***Example B*** "
            "or compare ***Example A*** with blank inputs \n"
            "4. Scroll down to see the comparison result")
st.markdown("### ***Your Example A***")
title_input_a = st.text_input("The title of Example A")
tag_input_a = st.text_input("The tags of Example A")
des_input_a = st.text_area("The description of Example A", height = 1)
img_a = st.file_uploader('Upload the thumbnail of Example A')

if img_a != None:
    #st.image(Image.open(img_a))
    thumbnail_a = Image.open(img_a).convert('RGB').resize((224,224))
    thumbnail_a = np.array([np.array(thumbnail_a)/255])
else:
    thumbnail_a = blank_image

need_b = st.selectbox('Would you like to enter inputs of Example B to compare?', [
    'No, I would like to compare Example A with blank inputs',
    'Yes, I would like to enter the inputs of Example B'
])

#page_B
if need_b == 'Yes, I would like to enter the inputs of Example B':
    st.markdown("### ***Your Example B***")
    title_input_b = st.text_input("The title of Example B")
    tag_input_b = st.text_input("The tags of Example B")
    des_input_b = st.text_area("The description of Example B", height = 1)
    img_b = st.file_uploader('Upload the thumbnail of Example B')
else:
    title_input_b = ""
    tag_input_b = ""
    des_input_b = ""

if img_b != None:
    #st.image(Image.open(img_b))
    thumbnail_b = Image.open(img_b).convert('RGB').resize((224,224))
    thumbnail_b = np.array([np.array(thumbnail_b)/255])
else:
    thumbnail_b = blank_image


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

#nlp_a
title_nlp_a = [w.lemma_.lower() for w in nlp(title_input_a) if not w.is_stop and not w.is_punct and not w.like_num]
title_a = [' '.join(title_nlp_a)]
tag_nlp_a = [w.lemma_.lower() for w in nlp(tag_input_a) if not w.is_stop and not w.is_punct and not w.like_num]
tag_a = [' '.join(tag_nlp_a)]
des_nlp_a = [w.lemma_.lower() for w in nlp(des_input_a) if not w.is_stop and not w.is_punct and not w.like_num]
des_a = [' '.join(des_nlp_a)]

test_title_a = title_tokenizer.texts_to_sequences(title_a)
test_title_a = sequence.pad_sequences(test_title_a, maxlen=maxlen_title)
test_tag_a = tag_tokenizer.texts_to_sequences(tag_a)
test_tag_a = sequence.pad_sequences(test_tag_a, maxlen=maxlen_tag)
test_des_a = des_tokenizer.texts_to_sequences(des_a)
test_des_a = sequence.pad_sequences(test_des_a, maxlen=maxlen_des)

#nlp_b
title_nlp_b = [w.lemma_.lower() for w in nlp(title_input_b) if not w.is_stop and not w.is_punct and not w.like_num]
title_b = [' '.join(title_nlp_b)]
tag_nlp_b = [w.lemma_.lower() for w in nlp(tag_input_b) if not w.is_stop and not w.is_punct and not w.like_num]
tag_b = [' '.join(tag_nlp_b)]
des_nlp_b = [w.lemma_.lower() for w in nlp(des_input_b) if not w.is_stop and not w.is_punct and not w.like_num]
des_b = [' '.join(des_nlp_b)]

test_title_b = title_tokenizer.texts_to_sequences(title_b)
test_title_b = sequence.pad_sequences(test_title_b, maxlen=maxlen_title)
test_tag_b = tag_tokenizer.texts_to_sequences(tag_b)
test_tag_b = sequence.pad_sequences(test_tag_b, maxlen=maxlen_tag)
test_des_b = des_tokenizer.texts_to_sequences(des_b)
test_des_b = sequence.pad_sequences(test_des_b, maxlen=maxlen_des)


#show table
if need_b == 'Yes':
    df = pd.DataFrame(columns=['Example A', 'Example B'], 
                      index=['Title', 'Tag', 'Description'])
    df.loc['Title'] = [title_input_a, title_input_b]
    df.loc['Tag'] = [tag_input_a, tag_input_b]
    df.loc['Description'] = [des_input_a, des_input_b]

    st.markdown("## Comparison")
    st.table(df)
    image_a, image_b = st.columns(2)
    with image_a:
        st.markdown('### Thumbnail A')
        st.image(thumbnail_a)
    with image_b:
        st.markdown('### Thumbnail B')
        st.image(thumbnail_b)

#predict
#baseline = model.predict([blank_image, test_fea, np.zeros((1,maxlen_title)), np.zeros((1,maxlen_tag)), np.zeros((1,maxlen_des))])[0][0]
prediction_a = model.predict([thumbnail_a, test_fea, test_title_a, test_tag_a, test_des_a])[0][0]
prediction_b = model.predict([thumbnail_b, test_fea, test_title_b, test_tag_b, test_des_b])[0][0]

#st.markdown(f'# {10 ** baseline}')
#st.markdown(f'# {10 ** prediction}')
#st.markdown(f'# Improvement: {int((10 ** prediction/ 10 ** baseline - 1) * 100)}%')

if need_b == 'Yes':

    if prediction_a > prediction_b: 
        st.markdown(f'# ***Example A*** is {int((10 ** (prediction_a - prediction_b) - 1) * 100)}% better than ***Example B***')
    elif prediction_a < prediction_b: 
        st.markdown(f'# ***Example B*** is {int((10 ** (prediction_b - prediction_a) - 1) * 100)}% better than ***Example A***')
    else:
        st.markdown('# ***Example A*** is as good as ***Example B***')


    st.markdown('* The number above shows the increasing percentage of views when comparing ***Example A*** with ***Example B***.\n'
                '* You can upload different thumbnails and enter different titles, tags and description to compare the scores of different combinations.')
else: 
    st.markdown(f'# Video Improvement Rate: {int((10 ** (prediction_a - prediction_b) - 1) * 100)}% \n'
                 '* The number above shows the increasing percentage of views compared to blank image, title, tag and description.\n'
                 '* You can upload different thumbnails and enter different titles, tags and description to compare the scores of different combinations.')
