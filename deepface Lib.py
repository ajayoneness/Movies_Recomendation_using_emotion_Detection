import cv2
from deepface import DeepFace
import numpy as np
import webbrowser
from collections import Counter



#function to search on google
def search_on_google(query):
    search_url = "https://www.google.com/search?q="
    query = query.replace(" ", "+")
    url = search_url + query
    webbrowser.open(url)




# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load emotion detection model
emotion_model = DeepFace.build_model('Emotion')

# Start webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

while True:
    print('''

    Movies By Emotion

    1. Open Camera
    2. About Us
    3. Contact US
    4. Exit

    ''')
    option = int(input("Select Option ----> "))

    if option == 1:
        num = 0
        emotion_list = []
        while True:
            ext = 0
            # Capture frame from webcam
            ret, frame = cap.read()

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            # Loop through detected faces
            for (x, y, w, h) in faces:
                
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

                # Extract face ROI for emotion detection
                face_roi = gray[y:y + h, x:x + w]

                # Resize face ROI for emotion model input
                face_roi = cv2.resize(face_roi, (48, 48))

                emotion_preds = emotion_model.predict(face_roi[np.newaxis, :, :, np.newaxis])

                emotion_label = emotion_labels[np.argmax(emotion_preds)]

                cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255   , 0), 2)


                if emotion_label != "neutral":
                    num = num+1
                    emotion_list.append(emotion_label)
                    word_freq = Counter(emotion_list)
                    most_frequent_word = word_freq.most_common(1)[0][0]
                    if num > 10:
                        search_on_google(f"suggest me movies name if my mood is {most_frequent_word}")
                        ext=1
                        print(most_frequent_word)



            if ext == 1:
                cv2.destroyAllWindows()
                break

            cv2.imshow('Live Emotion Detection', frame)

            # Break loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


    elif option==2:
        print('''
            About US
            -------------------------------------------
            I'm a Developer
        
        ''')

    elif option == 3:
        print('''
            Contact US
            ------------------------------------------
            Name :
            Collage : CIST

        
        ''')

    elif option == 4:
        exit()

    else:
        print("Invalid Option Selected !!")



# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
