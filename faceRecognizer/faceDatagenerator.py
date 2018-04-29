import cv2
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('../venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')

Id=input('enter your id:')
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,  scaleFactor=1.05, minNeighbors=5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("/home/sureshvairamuthu/Facial-Emotion-Detection/faceRecognizer/faceDataset/User."+Id +'.'+ str(sampleNum) + ".jpg", img[y:y+h,x:x+w])
        print("Added: ",Id+"."+str(sampleNum))
        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
