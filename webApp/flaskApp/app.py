#!/usr/bin/env python
from flask import Flask, render_template, Response, request
import cv2
import sqlite3
import os
import pygame as pg
import random
from predict import *


app = Flask(__name__, static_url_path='/static')


recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('/home/sureshvairamuthu/Facial-Emotion-Detection/faceRecognizer/faceTrainer/faceTrainer.yml')
recognizer.read('../../data/faceRecognizerData/faceTrainer/faceTrainer.yml')
# haar_face_cascade = cv2.CascadeClassifier('../venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
haar_face_cascade = cv2.CascadeClassifier('../../data/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

def getProfile(id):
    con = sqlite3.connect("../../data/faceRecognizerData/faceDatabase.db")
    cmd = "select * from knownPeople where ID = "+str(id)
    cursor = con.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
        #print(profile)
    con.close()
    return profile

def play_music(detection, volume=0.8):

    freq = 44100
    bitsize = -16
    channels = 2
    buffer = 2048
    pg.mixer.init(freq, bitsize, channels, buffer)
    pg.mixer.music.set_volume(volume)

    if detection =="sad":
        print("The person is not happy")
        song = random.randint(1,2)
        try:
            print("../data/musicFiles/happyFiles/{}.mp3".format(song))
            pg.mixer.music.load("../../data/musicFiles/happyFiles/{}.mp3".format(song))
            print("Music file loaded!")
        except pg.error:
            print("File not found! ({})".format( pg.get_error()))
            return
    elif (detection == "happy"):
        print("The person is happy or neutral")
        song = random.randint(1,3)
        try:
            print("../data/musicFiles/neutralFiles/{}.mp3".format(song))
            pg.mixer.music.load("../../data/musicFiles/neutralFiles/{}.mp3".format(song))
            print("Music file loaded")
        except pg.error:
            print("File not found! ({})".format( pg.get_error()))
            return
       

    
    clock = pg.time.Clock()
    # clock
    start_ticks = pg.time.get_ticks()
        
    pg.mixer.music.play()
    
    seconds = 0
    while pg.mixer.music.get_busy():
        seconds = (pg.time.get_ticks() - start_ticks) / 1000  # calculate how many seconds
        clock.tick(30)
        if seconds > 30:
            break

    #     print(seconds)

def faceGen():
    while True:

        ret, im =cam.read()

        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.2);
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            #print(gray[y:y+h,x:x+w].shape)
            #print(gray[y:y+h,x:x+w])
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            print("Conf=", conf)
            profile = getProfile(Id)
            if(profile != None):
                if(conf < 50.00):
                    cv2.putText(im, str(profile[1]), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                else:
                    cv2.putText(im, "Unknown", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            else:
                cv2.putText(im, "Unknown", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
        cv2.imwrite('face.jpg', im)
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('face.jpg', 'rb').read() + b'\r\n')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r



@app.route('/emotion')
def emotion():
    pred = detectEmotion('./static/emotion.jpg')

    emotion="None"
    print("pred value", pred, type(pred))
    if(pred == 0):
        print("The emotion is neutral and the person feels normal")
        emotion = "neutral"
    elif( pred == 1):
        print("The peson is angry")
        emotion = "angry"
    elif(pred == 2):
        print("The person feels contempt")
        emotion = "contempt"
    elif(pred == 3):
        print("The person feels disgust")
        emotion = "disgust"
    elif(pred == 4):
        print("The emotion is fear and the person is afraid")
        emotion = "afraid"
    elif(pred == 5):
        print("The person feels happy")
        emotion = "happy"
    elif(pred == 6):
        print("The person is sad")
        emotion = "sad"
    elif(pred == 7):
        print("The person is surprised")
        emotion = "surprised"

    if(pred == 1  or pred ==2 or pred == 3 or pred == 6):
        detection = "sad"
    elif(pred==0 or pred==4 or pred==5 or pred==7):
        detection = "happy"
    else:
        detection = "Neutral"
        pred = "None"
    return render_template('predict.html', output=pred, detection=detection)


@app.route('/play')
def play():
    song = request.args.get('song')
    if song == "Neutral":
        return "Emotion is Neutral, I cannot play any song!!!"
    play_music(song)
    return render_template('emotionRecog.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ret, im =cam.read()
        cv2.imwrite('./static/emotion.jpg', im)
        cv2.destroyAllWindows()
        return render_template('emotionRecog.html')
    else:
        return render_template('new.html')


@app.route('/video_feed')
def video_feed():
    return Response(faceGen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0')