
import cv2
import numpy as np
from PIL import Image
import os

path = 'Dataset_Faces'

cascadePath = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')
faceDetector = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.face.LBPHFaceRecognizer_create()

def training_function(path):
    images_dataset = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    number_faces = []
    tags = []
    
    for image_path in images_dataset:
        gray = Image.open(image_path).convert('L')  
        img_np = np.array(gray, 'uint8')
        
        face_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(img_np)
        
        for (x, y, w, h) in faces:
            number_faces.append(img_np[y:y + h, x:x + w])
            tags.append(face_id)
    
    return number_faces, tags

faces, tags = training_function(path)

if len(faces) > 0:
    recognizer.train(faces, np.array(tags))
    recognizer.write('model.yml')
    print(f"\n{len(np.unique(tags))} face(s) trained. Model saved as 'model.yml'")
else:
    print("\nNo faces found for training.")
