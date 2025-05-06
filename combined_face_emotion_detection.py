import cv2
import torch
import numpy as np
import platform
import pathlib
import os
import logging
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']  # Reordered to match number mapping
EMOTION_NUMBERS = {
    'angry': 1,
    'happy': 2,
    'neutral': 3,
    'sad': 4,
    'surprise': 5
}

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(current_dir, 'best.pt')  # Path to YOLOv5 model
EMOTION_MODEL_PATH = os.path.join(current_dir, 'CustomCNN_4.h5')  # Path to emotion model
HAAR_CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

IMG_SIZE = 100  # Size of input images for the emotion model
NUM_CLASSES = 5  # Number of emotion classes

# Global variables
yolo_model = None
emotion_model = None
face_cascade = None
dominant = None
aggregate_emotion = None
aggregate_confidence = None

# Handle path issues for Windows
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

def load_models():
    """
    Load both YOLOv5 face detection model and emotion classification model.
    Returns:
        tuple: (yolo_model, emotion_model)
    """
    global yolo_model, emotion_model, face_cascade
    
    try:
        # Load YOLOv5 model
        logger.info("Loading YOLOv5 model...")
        yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
        yolo_model.eval()
        logger.info("YOLOv5 model loaded successfully")
        
        # Load emotion model
        logger.info("Loading emotion model...")
        if not os.path.exists(EMOTION_MODEL_PATH):
            raise FileNotFoundError(f"Emotion model not found at {EMOTION_MODEL_PATH}")
        
        emotion_model = load_model(EMOTION_MODEL_PATH)
        logger.info("Emotion model loaded successfully")
        
        # Load Haar Cascade for face detection
        logger.info("Loading Haar Cascade classifier...")
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if face_cascade.empty():
            raise ValueError("Failed to load Haar Cascade classifier")
        logger.info("Haar Cascade classifier loaded successfully")
        
        return yolo_model, emotion_model
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def create_emotion_model():
    """Create the emotion model architecture matching the trained model"""
    model = Sequential()
    
    # First block
    model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', 
                    kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    # Second block
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    # Third block
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    # Fourth block
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def detect_faces(frame):
    """Detect faces using either YOLOv5 or Haar Cascade"""
    faces = []
    
    try:
        # Try YOLOv5 first
        if yolo_model is not None:
            results = yolo_model(frame)
            yolo_faces = results.xyxy[0]  # x1, y1, x2, y2, confidence, class
            
            for face in yolo_faces:
                x1, y1, x2, y2, conf, cls = face
                faces.append((int(x1), int(y1), int(x2), int(y2)))
            
            if faces:
                logger.debug(f"YOLOv5 detected {len(faces)} faces")
                return faces
    except Exception as e:
        logger.warning(f"YOLOv5 face detection failed: {str(e)}")
    
    # Fallback to Haar Cascade
    if face_cascade is not None:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            haar_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in haar_faces:
                faces.append((x, y, x + w, y + h))
            
            if faces:
                logger.debug(f"Haar Cascade detected {len(faces)} faces")
                return faces
        except Exception as e:
            logger.warning(f"Haar Cascade face detection failed: {str(e)}")
    
    logger.warning("No faces detected")
    return []

def preprocess_face(face_img):
    """Preprocess the face image for emotion prediction"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Resize to match model's expected sizing
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        # Convert to 3 channels (RGB)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # Preprocess the image
        gray = gray.astype('float32') / 255.0
        gray = np.expand_dims(gray, axis=0)
        return gray
    except Exception as e:
        logger.error(f"Error preprocessing face: {str(e)}")
        return None

def detect_emotion(frame, face_detector=None, emotion_model=None):
    """
    Detect emotions in the given frame using both YOLOv5 and emotion classification model.
    Args:
        frame: Input image frame
        face_detector: YOLOv5 model for face detection
        emotion_model: Emotion classification model
    Returns:
        str: Detected emotion
    """
    if face_detector is None or emotion_model is None:
        raise ValueError("Models not loaded. Call load_models() first.")
    
    try:
        # Convert frame to RGB for YOLOv5
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using YOLOv5
        results = face_detector(frame_rgb)
        faces = results.xyxy[0].cpu().numpy()
        
        if len(faces) == 0:
            return 'neutral'
        
        emotions = []
        confidences = []
        
        for face in faces:
            x1, y1, x2, y2, conf, _ = face
            if conf < 0.5:  # Confidence threshold
                continue
                
            # Extract face region
            face_img = frame[int(y1):int(y2), int(x1):int(x2)]
            if face_img.size == 0:
                continue
            
            # Preprocess face for emotion detection
            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            
            # Predict emotion
            predictions = emotion_model.predict(face_img, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = predictions[0][emotion_idx]
            
            emotions.append(EMOTIONS[emotion_idx])
            confidences.append(confidence)
        
        if not emotions:
            return 'neutral'
        
        # Get the most confident emotion
        max_conf_idx = np.argmax(confidences)
        return emotions[max_conf_idx]
        
    except Exception as e:
        logger.error(f"Error in emotion detection: {str(e)}")
        return 'neutral'

def get_dominant_emotion_number(predictions):
    """Convert emotion predictions to dominant emotion number"""
    global dominant
    emotion_idx = np.argmax(predictions)
    emotion = EMOTIONS[emotion_idx]
    dominant = EMOTION_NUMBERS[emotion]
    return dominant

def calculate_aggregate_emotion(all_predictions):
    """Calculate aggregate emotion from all face predictions"""
    global aggregate_emotion, aggregate_confidence
    
    if not all_predictions:
        aggregate_emotion = None
        aggregate_confidence = None
        return None, None
    
    # Convert predictions to numpy array if they aren't already
    predictions = np.array(all_predictions)
    
    # Count occurrences of each emotion
    emotion_counts = np.bincount(predictions, minlength=len(EMOTIONS)+1)[1:]  # Skip 0 index
    
    # Get the most frequent emotion
    max_count = np.max(emotion_counts)
    if max_count == 0:
        return None, None
    
    # Get all emotions that have the maximum count
    max_emotions = np.where(emotion_counts == max_count)[0] + 1  # +1 to match emotion numbers
    
    # If there's a tie, use the first one
    aggregate_emotion = max_emotions[0]
    aggregate_confidence = max_count / len(all_predictions)
    
    return aggregate_emotion, aggregate_confidence

def main():
    # Initialize webcam and load models
    cap = cv2.VideoCapture(0)
    load_models()  # Make sure this is called
    
    if not cap.isOpened():
        print("Unable to access the camera")
        return 'neutral'
    
    print("Starting combined face and emotion detection...")
    print("Press 'q' to quit")
    
    all_emotions = []
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect emotion
            emotion = detect_emotion(frame, yolo_model, emotion_model)
            emotion_number = EMOTION_NUMBERS[emotion]
            all_emotions.append(emotion_number)
            emotion_counts[emotion] += 1
            
            # Display current emotion
            cv2.putText(frame, f"Current Emotion: {emotion}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # Calculate and display aggregate emotion
            if len(all_emotions) > 0:
                agg_emotion, agg_confidence = calculate_aggregate_emotion(all_emotions)
                if agg_emotion is not None:
                    agg_text = f"Aggregate Emotion: {EMOTIONS[agg_emotion-1]} ({agg_emotion}): {agg_confidence:.2f}"
                    cv2.putText(frame, agg_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display emotion counts
            y_pos = 90
            for emotion, count in emotion_counts.items():
                if count > 0:
                    count_text = f"{emotion}: {count}"
                    cv2.putText(frame, count_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 30
            
            # Display the resulting frame
            cv2.imshow('Face and Emotion Detection', frame)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Get the final aggregate emotion
        if len(all_emotions) > 0:
            final_emotion, final_confidence = calculate_aggregate_emotion(all_emotions)
            if final_emotion is not None:
                print(f"\nFinal Aggregate Emotion: {EMOTIONS[final_emotion-1]} ({final_emotion}): {final_confidence:.2f}")
                return EMOTIONS[final_emotion-1]
        
    except Exception as e:
        print(f"Error during emotion detection: {str(e)}")
        return 'neutral'
    finally:
#######################################################################
    #     # Release resources
    #     cap.release()
    #     cv2.destroyAllWindows()
    
    # return 'neutral'
#############################################################
#Changed
        cap.release()
        cv2.destroyAllWindows()

        print(f"Aggregate Emotion: {aggregate_emotion}")
    return aggregate_emotion

if __name__ == "__main__":
    main()

##################################################################
# if __name__ == "__main__":
    # main() 