import cv2
import sqlite3
import os
import numpy as np


cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('../venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#detector=cv2.CascadeClassifier('../venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')


def insertOrUpdate(name):
	con = sqlite3.connect("faceDatabase.db")
	c= con.cursor()
	#cmd="SELECT ID,name FROM knownPeople where ID="+str(Id)
	#cursor = con.execute(cmd)
	#isRecordExist = 0
	#for row in cursor:
	#	isRecordExist =1
	#if(isRecordExist ==1):
	#	cmd = "update knownPeople set name ="+str(name)+" where ID="+str(Id)
	#else:
	cmd = "insert into knownPeople(Name) values("+str(name)+")"
	c.execute(cmd)

	Id=c.lastrowid
	#print(Id)
	con.commit()
	con.close()
	return Id



#Id=input('Enter ID:')
print("Enter name with \"\" ")
name=input('Enter your name:')

Id = insertOrUpdate(name)

sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,  scaleFactor=2.2, minNeighbors=5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("./faceDataset/User."+str(Id) +'.'+ str(sampleNum) + ".jpg", img[y:y+h,x:x+w])
        #cv2.imwrite("./faceDataset/User."+ str(Id) +'.'+ str(sampleNum) + ".jpg", img[y:y+h,x:x+w])

        print("Added: ",str(Id)+"."+str(sampleNum))
        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 50
    elif sampleNum>50:
        break

cam.release()
cv2.destroyAllWindows()



recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("../venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml");
#detector= cv2.CascadeClassifier("../venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml");


def getImagesAndLabels(path):
	#get the path of all the files in the folder
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
	#create empth face list
	faceSamples=[]
	#create empty ID list
	Ids=[]
	#now looping through all the image paths and loading the Ids and the images
	for imagePath in imagePaths:
		#loading the image and converting it to gray scale
		#image = #Image.open(imagePath).convert('L')
		image= cv2.imread(imagePath)
		pilImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#Now we are converting the PIL image into numpy array
		imageNp=np.array(pilImage,'uint8')
		#getting the Id from the image
		Id=int(os.path.split(imagePath)[-1].split(".")[1])
		#print(Id)
		# extract the face from the training image sample
		faces=detector.detectMultiScale(imageNp)
		#If a face is there then append that in the list as well as Id of it
		for (x,y,w,h) in faces:
			faceSamples.append(imageNp[y:y+h,x:x+w])
			Ids.append(Id)
	#print(Ids)
	return faceSamples,Ids


faces,Ids = getImagesAndLabels('./faceDataset')
#print(np.array(Ids))
#faces,Ids = getImagesAndLabels('./faceDataset')
recognizer.train(faces, np.array(Ids))
recognizer.save('./faceTrainer/faceTrainer.yml')
#recognizer.save('./faceTrainer/faceTrainer.yml')


