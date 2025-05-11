import cv2
import os
import time

# Load the face detection model
cascadePath = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')
faceDetector = cv2.CascadeClassifier(cascadePath)

# Initialize webcam (0 = default camera)
cam = cv2.VideoCapture(0)

# Set frame dimensions
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)

faceId = input("\nEnter Face ID (S.No): ")
print("\nLook at the camera.")
count = 0

dataset_path = "Dataset_Faces"
os.makedirs(dataset_path, exist_ok=True)

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        face_filename = os.path.join(dataset_path, f"Tag.{faceId}.{count}.jpg")
        cv2.imwrite(face_filename, gray[y:y + h, x:x + w])

    cv2.imshow("Face Capture", img)

    k = cv2.waitKey(100) & 0xFF
    if k == 27 or count >= 50:
        break

print("\nCapture complete. Exiting.")
cam.release()
cv2.destroyAllWindows()
