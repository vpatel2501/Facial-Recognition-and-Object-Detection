import cv2
import numpy as np
import time
import importlib.util
from threading import Thread
import matplotlib.pyplot as plt
from tensorflow.lite.python.interpreter import Interpreter

class Video_Webcam:
    def __init__(self, resolution=(640, 480), framerate=60):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.frame = None
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, self.frame = self.cap.read()
            if not ret:
                self.frame = None

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

labels, model_interpreter = None, None
input_details, output_details = None, None
height, width = None, None
floating_model, input_mean, input_std = None, 127.5, 127.5
imW, imH = 640, 480
minimum_confidence = 0.5

def load_model():
    global labels, model_interpreter, input_details, output_details
    global height, width, floating_model

    model_path = "detect.tflite"
    label_path = "labelmap.txt"

    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
        if labels[0] == '???':
            labels.pop(0)

    model_interpreter = Interpreter(model_path=model_path)
    model_interpreter.allocate_tensors()

    input_details = model_interpreter.get_input_details()
    output_details = model_interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = input_details[0]['dtype'] == np.float32

def detect_and_display(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    model_interpreter.set_tensor(input_details[0]['index'], input_data)
    model_interpreter.invoke()

    boxes = model_interpreter.get_tensor(output_details[0]['index'])[0]
    classes = model_interpreter.get_tensor(output_details[1]['index'])[0]
    scores = model_interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if minimum_confidence < scores[i] <= 1.0:
            ymin = int(max(1, boxes[i][0] * imH))
            xmin = int(max(1, boxes[i][1] * imW))
            ymax = int(min(imH, boxes[i][2] * imH))
            xmax = int(min(imW, boxes[i][3] * imW))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"{labels[int(classes[i])]}: {int(scores[i]*100)}%"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    return frame

def run_detection_mac():
    load_model()
    cam = Video_Webcam(resolution=(imW, imH)).start()
    time.sleep(2)

    plt.ion()
    fig, ax = plt.subplots()

    while True:
        frame = cam.read()
        if frame is None:
            continue

        frame = detect_and_display(frame)
        ax.clear()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        plt.pause(.04)

    cam.stop()
    plt.close()

if __name__ == "__main__":
    run_detection_mac()
