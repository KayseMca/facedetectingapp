import os
from flask import Flask, request, jsonify, render_template, url_for, redirect
from werkzeug.utils import secure_filename
from urllib.request import Request, urlopen

import cv2
import sys

# import tensorflow as tf
# from module.load import load_model
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot   as plt

from analysis_emotions import facecrop

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
UPLOAD_FOLDER = 'static/uploads/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


mood_message = [
    {
        "Happy":'Since you are happy, lets keep up the good mood with some amazing music!',
        "Sad":'It seems that you are having a bad day, lets cheer you up with some amazing music!',
        "Disgust":'It seems something has got you feeling disgusted. Lets improve your mood with some great music!',
        "Neutral":'It seems like a normal day. Lets turn it into a great one with some amazing music!',
        "Fear":'You seem very scared. We are sure that some music will help!',
        "Angry":'You seem angry. Listening to some music will surely help you calm down!',
        "Surprise":'You seem surprised! Hopefully its some good news. Lets celebrate it with some great music!'
    },
]

music = [
    {
        "Happy":'https://open.spotify.com/playlist/1BVPSd4dynzdlIWehjvkPj',
        "Sad":'https://www.writediary.com/ ',
        "Disgust":'https://open.spotify.com',
        "Neutral":'https://www.netflix.com/',
        "Fear":'https://www.youtube.com/watch?v=KWt2-lUpg-E',
        "Angry":'https://www.onlinemeditation.org/',
        "Angry":'https://www.onlinemeditation.org/',
        "Surprise":'https://brightside.me/wonder-curiosities/20-times-ordinary-things-surprised-us-when-we-least-expected-it-735510/'
    },
]
mood = [
    {
        "Happy":"success",
        "Angry":"danger",
        "Fear":"warning",
        "Neutral":"success",
        "Surprise":"success",
        "Disgust":"warning",
        "Sad":"info"
    }
    ]
acctivities = [
    {
        "Happy":'Try out some dance moves',
        "Sad":'• Write in a journal',
        "Disgust":'Listen soothing music',
        "Neutral":' Watch your favourite movie',
        "Fear":' Get a good sleep',
        "Angry":' Do meditation',
        "Surprise":' Give yourself a treat'
        },
]
response_data = [
    {
        "loading":True,
        "face_detect":None,
        "result":None,
        "mood_message":None,
        "music":None,
        "activities":None,
        "mood":"primary"
    }
]

# flask app init

app = Flask(__name__, static_url_path='/static')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET','POST'])
def home():
    """ Manual Uploading of Images via URL or Upload """

    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            response[0]['result'] = 'no image'
            return redirect(request.url)
        # img = request.files['file']

        img = request.files['file']
        if img.filename == '':
            response[0]['result'] = 'no image'
            return redirect(request.url)
        
        if img and allowed_file(img.filename):
            filename = secure_filename(img.filename)
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # predicting 
            data = facecrop(filename)
            if len(data)==1: 
                response = [
                    {
                        "face_detect":None,
                        "result"     :"Not detected any face"
                    }
                ]
                return render_template('notdetected.html', response=response, filename=filename)

            loading, circle, pred = data[0],data[1],data[2]
            response = response_data
            response[0]['loading'] = loading
            response[0]['face_detect'] = circle
            response[0]['result'] = pred
            response[0]['music'] = music[0][pred]
            response[0]['mood_message'] = mood_message[0][pred]
            response[0]['activities'] = acctivities[0][pred]
            response[0]['mood'] = mood[0][pred]
            # print(response)
    return render_template('success.html', response=response, filename=filename)

@app.route('/imageurl', methods=['POST'])
def imageurl():
    """ Fetches Image from URL Provided, does Emotion Analysis & renders."""

    # Fetch the Image from the Provided URL
    if request.form['url']=='':
        return redirect("/")
    url = request.form['url']
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})


    # Reading, Encoding and Saving it to the static Folder
    webpage = urlopen(req).read()
    arr = np.asarray(bytearray(webpage), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    path = "static/uploads/"
    cv2.imwrite(path + "urlimage.jpg", img)

    data = facecrop("urlimage.jpg")
    if len(data)==1: 
        response = [
            {
                "face_detect":None,
                "result"     :"Not detected any face"
            }
        ]

        return render_template('notdetected.html', response=response, filename="urlimage.png")

    loading, circle, pred = data[0],data[1],data[2]
    response = response_data
    response[0]['loading'] = loading
    response[0]['face_detect'] = circle
    response[0]['result'] = pred
    response[0]['music'] = music[0][pred]
    response[0]['mood_message'] = mood_message[0][pred]
    response[0]['activities'] = acctivities[0][pred]
    response[0]['mood'] = mood[0][pred]
    # print(response)
    return render_template('success.html', response=response, filename="urlimage.jpg")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(port= port, debug=True)
    