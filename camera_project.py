import cv2
import numpy as np7
import os

camera = cv2.VideoCapture(0)

while True:

    c,v = camera.read()

    v = cv2.cvtColor(v,cv2.COLOR_BGR2GRAY)
    path = os.path.dirname(cv2.__file__)

    face1 = cv2.CascadeClassifier(f"{path}/data/haarcascade_frontalface_default.xml")

    test = face1.detectMultiScale(v, scaleFactor=1.1,minNeighbors=5)

    print(test)
    for faca in test:
        x,y,w,h  =faca
        image1 = v[y:y+h,x:x+w]
        cv2.imwrite("Image.png",image1)
        cv2.rectangle(v,(x,y),(x+w,y+h),(255,255,255),thickness=4)

    cv2.imshow("Camera",v)
    cv2.waitKey(1)
