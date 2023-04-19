from django.shortcuts import render,redirect
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



def home(request):
    return render(request,'home.html')

def moviesList(request,emotion):


    if emotion == "angry":
        movlis =["Borat: Cultural Learnings of America for Make Benefit Glorious Nation of Kazakhstan",    "The Hangover",    "Bridesmaids",    "This Is Spinal Tap",    "Airplane!",    "The Princess Bride",    "The Grand Budapest Hotel",    "Dumb and Dumber",    "Groundhog Day",    "Anchorman: The Legend of Ron Burgundy"]

    elif emotion == "disgust":
        movlis = [    "Toy Story",    "Finding Nemo",    "The Lion King",    "Up",    "Moana",    "The Incredibles",    "Zootopia",    "Frozen",    "WALL-E",    "Ratatouille"]


    elif emotion == "fear":
        movlis = [    "The Princess Diaries",    "Mamma Mia!",    "Singin' in the Rain",    "The Greatest Showman",    "The Sound of Music",    "Mrs. Doubtfire",    "Legally Blonde",    "Elf",    "The Lion King",    "Toy Story"]


    elif emotion == "sad":
        movlis = [    "The Notebook",    "Titanic",    "A Walk to Remember",    "The Fault in Our Stars",    "500 Days of Summer",    "La La Land",    "Pretty Woman",    "Notting Hill",    "The Before Trilogy",    "Eternal Sunshine of the Spotless Mind"]


    elif emotion == "happy":
        movlis = [    "The Shining",    "The Exorcist",    "Halloween",    "A Nightmare on Elm Street",    "The Texas Chainsaw Massacre",    "Psycho",    "Rosemary's Baby",    "Hereditary",    "Get Out",    "The Conjuring"]


    elif emotion == "surprise":
        movlis = [    "Die Hard",    "The Matrix",    "Terminator 2: Judgment Day",    "Aliens",    "Predator",    "Lethal Weapon",    "Mission: Impossible - Fallout",    "Mad Max: Fury Road",    "John Wick",    "The Dark Knight"]

    else:
        movlis = ["Lov You MOM"]


    return render(request,'moviesList.html',{"emotion":emotion,"movlis":movlis})



def opencamera(request):
    num = 0
    emotion_list = []
    while True:
        ext = 0
        # Capture frame from webcam
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

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

            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if emotion_label != "neutral":
                num = num + 1
                emotion_list.append(emotion_label)
                word_freq = Counter(emotion_list)
                most_frequent_word = word_freq.most_common(1)[0][0]
                if num > 10:
                    # search_on_google(f"suggest me movies name if my mood is {most_frequent_word}")
                    ext = 1
                    print(most_frequent_word)
                    return redirect(f"/movieslist/{most_frequent_word}")


        if ext == 1:
            cv2.destroyAllWindows()
            break

        cv2.imshow('Live Emotion Detection', frame)

        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    return render(request,'home.html')




