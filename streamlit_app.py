import streamlit as st
import pickle
import datetime as dt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import keras
import spacy


#nlp = spacy.load('en_core_web_sm')
@st.cache(allow_output_mutation=True)
def spacy_load(model_name):
    nlp = spacy.load(model_name)
    return nlp
nlp = spacy_load('en_core_web_sm')

#with open('pickles/dl_5.h5', 'rb') as f:
#    model = keras.models.load_model(f)
@st.cache(allow_output_mutation=True)
def load_model(path):
    model = keras.models.load_model(path)
    return model
model = load_model('pickles/dl_5.h5')

#with open('pickles/feature_scaler_1213.pkl', 'rb') as f:
#    scaler = pickle.load(f)
#with open('pickles/title_tokenizer_1213.pkl', 'rb') as f:
#     title_tokenizer = pickle.load(f)
# with open('pickles/tag_tokenizer_1213.pkl', 'rb') as f:
#     tag_tokenizer = pickle.load(f)
# with open('pickles/des_tokenizer_1213.pkl', 'rb') as f:
#     des_tokenizer = pickle.load(f)
@st.cache(allow_output_mutation=True)
def load_parameters(path):
    with open(path, 'rb') as f:
        parameters = pickle.load(f)
        return parameters
scaler = load_parameters('pickles/feature_scaler_1213.pkl')
title_tokenizer = load_parameters('pickles/title_tokenizer_1213.pkl')
tag_tokenizer = load_parameters('pickles/tag_tokenizer_1213.pkl')
des_tokenizer = load_parameters('pickles/des_tokenizer_1213.pkl')

#features = ['contentDetails.duration', 'status.madeForKids_x', 'hd','rectangular', 'video_publish_sec', 'snippet.categoryId_1', 'snippet.categoryId_10', 
#            'snippet.categoryId_17', 'snippet.categoryId_19', 'snippet.categoryId_2', 'snippet.categoryId_20', 'snippet.categoryId_22', 'snippet.categoryId_23',
#            'snippet.categoryId_24', 'snippet.categoryId_25', 'snippet.categoryId_26', 'snippet.categoryId_27', 'snippet.categoryId_28', 'snippet.categoryId_29', 
#            'statistics.subscriberCount', 'statistics.videoCount', 'status.madeForKids_y', 'US', 'IN', 'GB', 'CA', 'AU', 'channel_publish_sec']

today = dt.date.today()
categories = ['Entertainment', 'Gaming', 'Film & Animation', 'Sports', 'News & Politics', 'People & Blogs', 'Howto & Style', 'Music',
              'Autos & Vehicles', 'Education', 'Travel & Events', 'Science & Technology', 'Comedy', 'Nonprofits & Activism', 'Other']



#sidebar
st.sidebar.markdown('# YouTube Views Predcition System')
st.sidebar.markdown('## Video Information')
st.sidebar.write('1. The duration of your video:')
day = st.sidebar.number_input("Days", min_value = 0)
hour = st.sidebar.number_input("Hours", min_value = 0)
minute = st.sidebar.number_input("Minutes", min_value = 0, value = 5)
second = st.sidebar.number_input("Seconds", min_value = 0)
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
st.markdown('# You can use this system to optimize your thumbnail, title, tag, and description. Please fill out the information on the sidebar first.')
title_input = st.text_input("The title of your video")
tag_input = st.text_input("The tags of your video")
des_input = st.text_area("The description of your video", height = 1)
img = st.file_uploader('Upload your thumbnail')

if img != None:
    st.image(Image.open(img))
    thumbnail = Image.open(img).convert('RGB').resize((224,224))
    thumbnail = np.array([np.array(thumbnail)/225]) #225 is typo in training, need to change to 255 if retrain
else:
    thumbnail = np.zeros((1,224,224,3))


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

#nlp
title_nlp = [w.lemma_.lower() for w in nlp(title_input) if not w.is_stop and not w.is_punct and not w.like_num]
title = [' '.join(title_nlp)]
tag_nlp = [w.lemma_.lower() for w in nlp(tag_input) if not w.is_stop and not w.is_punct and not w.like_num]
tag = [' '.join(tag_nlp)]
des_nlp = [w.lemma_.lower() for w in nlp(des_input) if not w.is_stop and not w.is_punct and not w.like_num]
des = [' '.join(des_nlp)]


test_title = title_tokenizer.texts_to_sequences(title)
test_title = sequence.pad_sequences(test_title, maxlen=23)
test_tag = tag_tokenizer.texts_to_sequences(tag)
test_tag = sequence.pad_sequences(test_tag, maxlen=105)
test_des = des_tokenizer.texts_to_sequences(des)
test_des = sequence.pad_sequences(test_des, maxlen=855)

#predict
predictions = []
months = [1, 3, 6, 9, 12]
for time in months:
    features = [duration, vidoe_kid_yes, hd_yes, rectangular_yes, 60*60*24*30*time, categoryId_1, categoryId_10, 
                categoryId_17, categoryId_19, categoryId_2, categoryId_20, categoryId_22, categoryId_23,
                categoryId_24, categoryId_25, categoryId_26, categoryId_27, categoryId_28, categoryId_29, 
                subscriber, channel_video, channel_kid_yes, US, IN, GB, CA, AU, channel_duration]
    test_fea = scaler.transform(np.array(features).reshape(1, -1))
    prediction = model.predict([thumbnail, test_fea, test_title, test_tag, test_des]) 
    predictions.append(int(prediction[0][0]))

st.set_option('deprecation.showPyplotGlobalUse', False)
chart = sns.lineplot(x=months, y=predictions, marker = "o")
chart.set(ylim = (0, max(predictions)*1.1))
chart.set_xticks(months)
sns.despine()
for i, label in enumerate (predictions):
    plt.annotate(label, (months[i]-0.5, predictions[i]))
st.pyplot()


