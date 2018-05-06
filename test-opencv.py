import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Mac Path
haar_face_cascade = cv2.CascadeClassifier('./venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Ubuntu Path is different
# haar_face_cascade = cv2.CascadeClassifier('./venv/lib/python3.5/site-packages/cv2/data/haarcascade_frontalface_default.xml')


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5);
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)



    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
