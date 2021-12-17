import requests, io, json, time, datetime
from pymongo import MongoClient

api_key='*****************************'

connection_string = '*******************************'
client = MongoClient(connection_string)

db = client.youtube
videos = db.videos
channels = db.channels

def search_list(category_id, pages):
    '''
    Given category id and the number of pages and return the list of video id and channel id
    '''
    url='https://www.googleapis.com/youtube/v3/search'
    pageToken = ''
    params = {'key': api_key, 'part': 'snippet', 'maxResults': 50, 'relevanceLanguage': 'en' ,
              'type': 'video', 'videoCategoryId': str(category_id), 'pageToken': pageToken}
    video_id = []
    channel_id = []
    for p in range(pages):
        req=requests.get(url, params=params)
        js = json.loads(req.text)
        for v in js['items']:
            video_id.append(v['id']['videoId'])
            channel_id.append(v['snippet']['channelId'])
        if 'nextPageToken' in js:
            pageToken = js['nextPageToken']
            params = {'key': api_key, 'part': 'snippet', 'maxResults': 50, 'relevanceLanguage': 'en' ,
                      'type': 'video', 'videoCategoryId': str(category_id), 'pageToken': pageToken}
        else:
            break
    return video_id, channel_id

def save_video_data(video_id_list, timestamp):
    '''
    Given video ids and timestamp and insert the data into MongoDB Atlas
    '''
    part = ['contentDetails', 'id', 'snippet', 'statistics', 'status', 'topicDetails']
    url = 'https://www.googleapis.com/youtube/v3/videos'
    for t in range(0, len(video_id_list), 50):
        params = {'key': api_key, 'part': part, 'id' : video_id_list[t:t+50]}
        reg = requests.get(url, params=params)
        video_js = json.loads(reg.text)['items']
        for doc in video_js:
            doc.update({'timestamp': timestamp})
        videos.insert_many(video_js)
    pass


def save_channel_data(channel_id_list, timestamp):
    '''
    Given channel ids and timestamp and insert the data into MongoDB Atlas
    '''
    part = ['id', 'localizations', 'snippet', 'statistics', 'status']
    url = 'https://www.googleapis.com/youtube/v3/channels'
    for t in range(0, len(channel_id_list), 50):
        params = {'key': api_key, 'part': part, 'id' : channel_id_list[t:t+50]}
        reg = requests.get(url, params=params)
        channel_js = json.loads(reg.text)['items']
        for doc in channel_js:
            doc.update({'timestamp': timestamp})
        channels.insert_many(channel_js)
    pass

#the only function will run again in lambda
def lambda_handler(event, context):
    v_list = []
    c_list = []
    for cat in [24, 20, 1, 17, 25, 22, 26, 10, 2, 27, 19, 28, 23, 29]:
        v_temp, c_temp = search_list(cat, 2) #2 pages
        v_list += v_temp
        c_list += c_temp
    now = datetime.datetime.utcnow().isoformat()
    save_video_data(list(set(v_list)), now)
    save_channel_data(list(set(c_list)), now)
    return "done"