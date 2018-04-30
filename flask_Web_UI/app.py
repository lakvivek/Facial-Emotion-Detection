#!/usr/bin/env python
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('/home/sureshvairamuthu/Facial-Emotion-Detection/faceRecognizer/faceTrainer/faceTrainer.yml')
recognizer.read('../faceRecognizer/faceTrainer/faceTrainer.yml')
# haar_face_cascade = cv2.CascadeClassifier('../venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
haar_face_cascade = cv2.CascadeClassifier('../venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

Id=0

vc = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')


def gen():
    while True:
        rval, frame = vc.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=2.2)
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
            print(gray[y:y+h,x:x+w].shape)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            if(conf<50):
                if(Id==1):
                    name="Sri Teja"
                elif(Id==2):
                    name="Vivek"
            else:
                name="Unknown"
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.imwrite('t.jpg',frame) 
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)