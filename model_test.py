import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('emotion_detection_model.h5')

# Define the emotion categories corresponding to the output classes of the model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# Function to preprocess the frame (crop, resize, normalize)
def preprocess_image(image, target_size=(48, 48)):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the grayscale image to the target size (48x48)
    gray_resized = cv2.resize(gray, target_size)
    
    # Normalize the pixel values (scale them between 0 and 1)
    normalized_image = gray_resized / 255.0
    
    # Reshape to match the input shape for the CNN: (1, 48, 48, 1)
    return np.expand_dims(np.expand_dims(normalized_image, axis=-1), axis=0)



# Start the video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break
    
    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame using the face cascade
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) - the face
        roi = frame[y:y+h, x:x+w]

        # Preprocess the face ROI for the CNN model
        preprocessed_face = preprocess_image(roi)

        # Predict the emotion using the pre-trained model
        emotion_prediction = model.predict(preprocessed_face)
        
        # Get the emotion with the highest probability
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]

        # Display the predicted emotion on the frame
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
