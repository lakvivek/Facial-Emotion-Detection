from __future__ import division
import cv2
import numpy as np
import os
import gzip
import tarfile
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as ss
import matplotlib.image as img
import tensorflow as tf
from sklearn.utils import shuffle
import cv2
from modelRestorer import *
from datetime import datetime


Ws, Bs = initialise_variables()
W_Rs, B_Rs = restoremodel(Ws, Bs)

# haar_face_cascade = cv2.CascadeClassifier('../venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
haar_face_cascade = cv2.CascadeClassifier('../../venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def processInput(box):
	box = np.asarray(box, dtype=np.float32)
	box = box.reshape(1, 256, 256, 1)
	return box



def detectEmotion(filename):
	img = cv2.imread(filename)

	gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=2.2);
	for(x,y,w,h) in faces:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
	    gray_box = gray[y:y+h,x:x+w]
	    print(gray_box.shape[0], gray_box.shape[1])
	    if gray_box.shape[0] == 256 and gray_box.shape[1] == 256:
	    	start = datetime.now()
	    	arr = processInput(gray_box)
	    	pred = predict_model(W_Rs, B_Rs, arr)
	    	print("prediction is %s "%pred)
	    	print(datetime.now() - start)
	    else:
        	print("Image size is small")

	cv2.imshow('dst_rt', img)
	cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	detectEmotion('../../data/emotionRecognizerData/test1.png')

