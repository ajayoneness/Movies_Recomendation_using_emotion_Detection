import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained emotion detection model
emotion_model = load_model('path_to_your_emotion_model.h5')

# Define emotion labels and corresponding emojis
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
emoji_mapping = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😃',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

# Open camera capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pre-process frame for emotion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    gray = gray / 255.0
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)

    # Predict emotion from pre-processed frame
    prediction = emotion_model.predict(gray)
    emotion_index = np.argmax(prediction)
    emotion_label = emotion_labels[emotion_index]
    emoji = emoji_mapping[emotion_label]

    # Overlay emoji on frame
    cv2.putText(frame, emoji, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all windows
cap.release()
cv2.destroyAllWindows()
