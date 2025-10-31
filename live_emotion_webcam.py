import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Try to set stdout encoding to utf-8 to avoid UnicodeEncodeError on Windows consoles
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    # sys.stdout may not support reconfigure in some environments; ignore if it fails
    pass

# Load the trained model
print("Loading model...")
model = load_model("emotion_recognition_CNN.h5")

print("Model loaded!")

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
print("Starting webcam...")
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not access webcam!")
    exit()

print("Webcam started. Press 'q' to quit.")
print("="*60)

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Prepare for prediction
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        
        # Predict emotion
        prediction = model.predict(roi, verbose=0)
        label_index = np.argmax(prediction)
        predicted_emotion = emotion_labels[label_index]
        confidence = np.max(prediction) * 100
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display emotion and confidence
        cv2.putText(frame, f"{predicted_emotion} ({confidence:.1f}%)", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Emotion Detection - Press 'q' to quit", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
print("\n✅ Webcam closed!")
