#!/usr/bin/env python
from flask import Flask, render_template, Response, request
import cv2
import sqlite3
import os
from predict import *

app = Flask(__name__)


recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('/home/sureshvairamuthu/Facial-Emotion-Detection/faceRecognizer/faceTrainer/faceTrainer.yml')
recognizer.read('../../data/faceRecognizerData/faceTrainer/faceTrainer.yml')
# haar_face_cascade = cv2.CascadeClassifier('../venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
haar_face_cascade = cv2.CascadeClassifier('../../venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
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
    
        cv2.imwrite('face.jpg', im)
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('face.jpg', 'rb').read() + b'\r\n')

@app.route('/face', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ret, im =cam.read()
        cv2.imwrite('./static/emotion.jpg', im)
        cv2.destroyAllWindows()
        return render_template('emotionRecog.html')
    else:
        return render_template('faceRecog.html')


@app.route('/emotion')
def emotion():
    pred = detectEmotion('./static/emotion.jpg')
    return render_template('predict.html', output=pred)




@app.route('/video_feed')
def video_feed():
    return Response(faceGen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)