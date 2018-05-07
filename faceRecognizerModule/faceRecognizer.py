import cv2
import numpy as np
import sqlite3

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('../data/faceRecognizerData/faceTrainer/faceTrainer.yml')
#recognizer.read('./faceTrainer/faceTrainer.yml')
# haar_face_cascade = cv2.CascadeClassifier('../venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
haar_face_cascade = cv2.CascadeClassifier('../venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def getProfile(id):
	con = sqlite3.connect("../data/faceRecognizerData/faceDatabase.db")
	cmd = "select * from knownPeople where ID = "+str(id)
	cursor = con.execute(cmd)
	profile = None
	for row in cursor:
		profile = row
		#print(profile)
	con.close()
	return profile


cam = cv2.VideoCapture(0)


while True:
	ret, im =cam.read()
	#print

	gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

	faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.2);
	for(x,y,w,h) in faces:
		cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
		#print(gray[y:y+h,x:x+w].shape)
		#print(gray[y:y+h,x:x+w])
		Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
		#print("Id=",Id)
		#
		# cnf = int(conf)
		print("Conf=", conf)	
		profile = getProfile(Id)
		if(profile != None):
			if(conf < 50):
				cv2.putText(im, str(profile[1]), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
			else:
				cv2.putText(im, "Unknown", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
		
	cv2.imshow('im',im) 
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cam.release()
cv2.destroyAllWindows()
