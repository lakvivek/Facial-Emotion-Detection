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
    

def gen():
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
            print("Conf=", conf)
            profile = getProfile(Id)
            if(profile != None):
                if(conf < 50.00):
                    cv2.putText(im, str(profile[1]), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                else:
                    cv2.putText(im, "Unknown", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.imwrite('face.jpg',im)
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('face.jpg', 'rb').read() + b'\r\n')