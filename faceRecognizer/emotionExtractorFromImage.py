import cv2
import numpy as np

# haar_face_cascade = cv2.CascadeClassifier('../venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
haar_face_cascade = cv2.CascadeClassifier('../venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')




img = cv2.imread('sample.png')

gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=2.2);
for(x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
	print(gray[y:y+h,x:x+w].shape)
	print(gray[y:y+h,x:x+w])

cv2.imshow('dst_rt', img)
cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)

cv2.waitKey(0)
cv2.destroyAllWindows()
