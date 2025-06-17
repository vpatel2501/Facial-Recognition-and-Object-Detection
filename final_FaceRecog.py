import cv2
import numpy as np
import os
import threading

cascadePath = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')
faceDetector = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('model.yml')

font = cv2.FONT_HERSHEY_SIMPLEX

name_data = ['none', 'Veeren', 'Bruno']

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def face_recognition(dummy1, dummy2):
    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            
            if confidence < 100:
                name = name_data[id] if id < len(name_data) else "Unknown"
                confidence_text = " {0}%".format(round(100 - confidence))
            else:
                name = "Unknown"
                confidence_text = " {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('Facial Recognition', img)

        if cv2.waitKey(10) & 0xFF == 27:
            break

    print("\nExiting the program.")
    cam.release()
    cv2.destroyAllWindows()

face_recognition(None, None)
