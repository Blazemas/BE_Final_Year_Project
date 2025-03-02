import cv2
import torch
import numpy as np
from keras.models import model_from_json
import platform
import pathlib
from keras.models import Sequential

from keras.models import Sequential, model_from_json
from keras.saving import register_keras_serializable

from tensorflow.keras.models import Sequential, model_from_json


@register_keras_serializable()
class CustomSequential(Sequential):
    pass


# Fix pathlib issue on Windows
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# Load YOLOv5 model for face detection
face_model_path = pathlib.Path("D:/coding/BE Project/Music Recommendation System/face_expressions/Yolo Trial 1/best.pt").resolve()
face_model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(face_model_path))

# Load emotion detection model
json_file = open("facialemotionmodel.json", "r")  # Update path
model_json = json_file.read()
json_file.close()
# emotion_model = model_from_json(model_json)
emotion_model = model_from_json(model_json, custom_objects={"Sequential": CustomSequential})

emotion_model.load_weights("facialemotionmodel.h5")  # Update path

# Function to preprocess face for emotion model
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to access the camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Perform face detection using YOLOv5
    results = face_model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Get bounding boxes
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det  # Bounding box coordinates
        face = frame[int(y1):int(y2), int(x1):int(x2)]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (48, 48))
        img = extract_features(gray_face)
        pred = emotion_model.predict(img)
        prediction_label = labels[pred.argmax()]
        
        # Draw bounding box and emotion label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, prediction_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Display the result
    cv2.imshow('Face & Emotion Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()