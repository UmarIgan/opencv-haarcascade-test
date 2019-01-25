import numpy as np
import cv2
import pafy
url="youtube test video url"
videopafy=pafy.new(url)
best=videopafy.getbest(preftype="webm")

full_body=cv2.CascadeClassifier(path) #fullbody haarcascade path
upper_body=cv2.CascadeClassifier(path) #upperbody haarcascade path
face_haar=cv2.CascadeClassifier(path) #face haarcascade path
cap=cv2.VideoCapture(best.url)
while True:
    ret, img=cap.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    BODY=full_body.detectMultiScale(gray, 1.01, 8)#to get better result play with parameters here.
    
    for (x, y, w, h) in BODY:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        UP=upper_body.detectMultiScale(gray, 1.02, 4)
        for (ex, ey, ew, eh) in UP:
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            FACE=face_haar.detectMultiScale(gray, 1.3, 5)
            for (ax, ay, aw, ah) in FACE:
                cv2.rectangle(img, (ax, ay), (ax+aw, ay+ah), (0, 255, 0), 2)
            
    cv2.imshow('img', img)
    k=cv2.waitKey(30) & 0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
