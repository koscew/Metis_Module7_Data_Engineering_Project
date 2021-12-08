### Project Proposal of Data Engineering Project
Chien Yuan Chang
#### Question/need:
I am going to use the data of YouTube Videos obtained by YouTube Data API to build a model to predict the views and deploy a web app for YouTubers to estimate the views of their video and make adjustments of their title, description, tags, thumbnail, etc.

>* What is the framing question of your analysis, or the purpose of the model/system you plan to build?
>* Who benefits from exploring this question or building this model/system?

#### Data Description:
* Method: I will use YouTube Data API to search the list of certain categories and types of videos and get the detailed information of each video. 
* Size: I will be able to get the data of at most 4,000 videos per day with the default quota of YouTube Data API. It will be about 28,000 data points for the first week if there no other limits
* Sample of features expected to work with
    * videoId: '-eU9kqXcLVM', id of the video
    * channelId: 'UCqmld-BIYME2i_ooRTo1EOg', id of the channel for getting the statistics of the channel
    * publishedAt: '2021-12-06T07:33:11Z', the published time
    * thumbnail: 120 x 90 image, scraped separately from 'https://i.ytimg.com/vi/{videoId}/default.jpg'
    * title: 'New Skin |Selena "Lady Vengeance" | Mobile Legends: Bang Bang', the title of the video
    * description: The document of full description of the video
    * viewCount: 846750, the number of views of the video
    * duration: 'PT1M36S', the ISO 8601 duration of the video
    * tags: ['MobileLegends'], a list of the tags of the videos
    * dimension: '2d', whether the video is available in 3D or in 2D
    * dimension: 'hd', whether the video is available in high definition (HD) or only in standard definition(SD)
    * caption: false, whether captions are available for the video
    * categoryId: 20, the YouTube video category associated with the video
    * embeddable: True, whether the video can be embedded on another website
    * madeForKids: False, whether the video is designated as child-directed
    * subscriberCount: 13300000, the number of subscribers of the channel
    * videoCount: 1775, the number of videos of the channel

 
>* What dataset(s) do you plan to use, and how will you obtain the data?
>* What is an individual sample/unit of analysis in this project? What characteristics/features do you expect to work with?
>* If modeling, what will you predict as your target?

#### Tools:
* Python Requests and Json for data acquisition
* MongoDB/AWS for local/cloud database
* Python Pandas and Numpy for data clean, data restructuring, exploratory data analysis and feature engineering
* Python Scikit-learn for linear model
* Python Keras and TensorFlow for neural network model
* Python Streamlit/Flask for web applications and model deployment
* Python Matplotlib and Python Seaborn for data visualization
* Other Python libraries or tools if needed

>* How do you intend to meet the tools requirement of the project? 
>* Are you planning in advance to need or use additional tools beyond those required?

#### MVP Goal:
* A baseline model and working application on the local machine

>* What would a minimum viable product (MVP) look like for this project?
