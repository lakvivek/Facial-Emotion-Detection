import cv2
import sqlite3


cam = cv2.VideoCapture(0)
# detector=cv2.CascadeClassifier('../venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')
detector=cv2.CascadeClassifier('../venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')


def insertOrUpdate(name):
	con = sqlite3.connect("../data/faceRecognizerData/faceDatabase.db")
	c= con.cursor()
	#cmd="SELECT ID,name FROM knownPeople where ID="+str(Id)
	#cursor = con.execute(cmd)
	#isRecordExist = 0
	#for row in cursor:
	#	isRecordExist =1
	#if(isRecordExist ==1):
	#	cmd = "update knownPeople set name ="+str(name)+" where ID="+str(Id)
	#else:
	cmd = "INSERT INTO knownPeople(Name) VALUES('" + str(name) + "');"
	#print(cmd)
	c.execute(cmd)

	Id=c.lastrowid
	#print(Id)
	con.commit()
	con.close()
	return Id



#Id=input('Enter ID:')
#print("Enter name with \"\" ")
name=input('Enter your name:')

Id = insertOrUpdate(name)
print("PLEASE BE PATIENT UNTIL SYSTEM CAPTURES YOUR FACE IMAGE")
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,  scaleFactor=1.2);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("../data/faceRecognizerData/faceDataset/User."+str(Id) +'.'+ str(sampleNum) + ".jpg", img[y:y+h,x:x+w])
        #cv2.imwrite("./faceDataset/User."+ str(Id) +'.'+ str(sampleNum) + ".jpg", img[y:y+h,x:x+w])

        #print("Added training pic: ",sampleNum)
        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 50
    elif sampleNum>50:
        break

cam.release()
cv2.destroyAllWindows()



