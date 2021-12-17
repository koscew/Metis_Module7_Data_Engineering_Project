# YouTube Video Information Evaluation System
Chien Yuan Chang

## Abstract
The goal of this project is to build a data pipeline to ingest the information of YouTube videos by YouTube Data API and store in a database, build a model to predict the views, and deploy a web app for YouTuber creators to optimize the titles, descriptions, tags, thumbnail, etc.

I used [YouTube Data API](https://developers.google.com/youtube/v3/docs/) to search the list of the videos among 14 most common categories and get the detailed information of each video and its channel. Amazon Lambda with EventBridge as trigger was used to request the list and the detailed information of the videos and channels of 1,400 videos among 14 categories by YouTube Data API automatically every 8 hours and store the data into MongoDB Atlas. Then, I used Google Colab to requested the images of thumbnails by url and video id and stored the images into the Google Drive. 

I built a neural network model with five branches. The MSE and MAE were 1.08 and 0.79 on validation data and 1.08 and 0.78 on test data. Then, I deployed a web app with this model on [streamlit.io](https://share.streamlit.io/koscew/metis_module7_data_engineering_project/main) for users to upload different thumbnails and enter different titles, tags and descriptions to compare the scores of different combinations to optimize their YouTube video information and thumbnails.

## Design
5 billion videos are watched on Youtube every single day, 300 Hours of video are uploaded every minute, and every 1000 video views can earn 3 to 5 dollars. More view can bring more revenue. Building a web App to estimate the views of the video can help YouTube creators optimize their YouTube video information such as titles, tags, descriptions and thumbnails.

## Data
I used [YouTube Data API](https://developers.google.com/youtube/v3/docs/) to search the list of the videos among 14 most common categories and get the detailed information of each video and its channel. Amazon Lambda with EventBridge as trigger was used to request the list and the detailed information of the videos and channels of 1,400 videos among 14 categories by YouTube Data API automatically every 8 hours and store the data into MongoDB Atlas. Then, I used Google Colab to requested the images of thumbnails by url and video id and stored the images into the Google Drive.  

There were Total 10,472 unduplicated and clean data points were split into 6,282(60%) for training, 2,095(20%) for validation and 2,095(20%)for test.

The features I worked on were:  

- Numerical and Categorical Video Information
    - View count
    - Published time
    - Duration
    - Category
    - Definition
    - Dimension
    - Made for kids
    - Subscriber count
    - Channel video count
- Images
    - Thumbnails
- Text
    - Title
    - Tags
    - Description

## Algorithms

***Amazon Lambda Function***

1. Requested YouTube Data API to get random videos in English among 14 common categories 
2. Requested YouTube Data API to get video information of the videos, added timestamp,  and insert into MongoDB Atlas
3. Requested YouTube Data API to get channel information of the channels, added timestamp, and insert into MongoDB Atlas 
4. Used the trigger feature in Amazon Lambda with EventBridge to set a timer to run the code every 484 minutes which considered the average 35-second execution time to prevent the error of running out the quota.

***Feature Engineering***

- Transferred published time and duration to datetime format
- Subtract published time from the timestamp of the API request to get the published period
- Divided the view count by published period to get average daily views
- Logged average daily views by 10 as the target
- One-hot encoded and/or standardized the numerical and categorical features
- Text preprocessing and tokenizing of text data


***Baseline Model - Linear Regression***

- Numerical and Categorical Video Information was one-hot encoded and/or standardized
- Thumbnails were resized and rescaled with TruncatedSVD
- Titles, tags and descriptions went through NLP, tokenizer and PCA
- The MSE and MAE of baseline model with numerical and categorical features of validation data were 1.57 and 0.96
- The MSE and MAE were larger when combining with image and text data

***Final Model - Neural Network***

- 5 branches of data
    - Numerical and Categorical Data went through two dense layers
    - Image data used transfer learning with MobileNetV2 and flattened to connect with a dense layer
    - Title data used embedding and flattened to connect with a dense layer
    - Tag data used embedding and flattened to connect with a dense layer
    - Description data used embedding, convolution1D, 
maxPooling1D, and Bidirectional LSTM to connect with a dense layer
- Concatenate 5 branches at last and went through another two dense layer and one dropout layer
- The MSE and MAE of validation data were 1.08 and 0.79 and of test data were 1.08 and 0.78

***Application Deploy***
  
Streamlit was used to build the application and visualize the recommendation system. The scaler of the numerical and categorical features, tokenizers of titles, tags and descriptions, and the neural network model were packed and loaded by pickle. The users can 
input video information, channel information, titles, tags, descriptions to get a score which will compare with blank inputs to optimize their inputs. 

## Tools
- Amazon Lambda for running code automatically and periodically on the cloud
- MongoDB for data storage
- Python Pymongo for data management
- Python Pandas and Numpy for data clean, data restructuring, exploratory data analysis and feature engineering
- Python spaCy for text processing
- Python Scikit-learn for standardization, vectorization, dimensionality reduction, and simple linear regression
- Python Pillow for image preprocessing
- Google Colab for GPU boosting
- Python Keras and TensorFlow for neural network model
- Python Matplotlib, Seaborn for data visualization
- Python Pickle for packaging the models, matrices and data
- Python Streamlit for visualization and web application

## Communication
In addition to [the slides of the final presentation](final_presentation.pdf), [charts](images/), [codes](codes/) and this written description, the evaliuation system was deployed on [Streamlit](https://share.streamlit.io/koscew/metis_module7_data_engineering_project/main) and the findings will also be posted on [my personal blog](https://koscew.github.io/) in the future.
