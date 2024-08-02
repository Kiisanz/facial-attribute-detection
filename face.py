import cv2
import numpy as np

# Load pre-trained models for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load models for age and gender prediction
age_model = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
gender_model = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

# Define the list of age and gender labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(14-17)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Define colors in BGR format
frame_color = (0, 255, 255)  # Green for the face rectangle
text_color = (255, 255, 255)  # Red for the text

resize_width = 426
resize_height = 240

def detect_attributes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Prepare blob for age and gender
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0))
        age_model.setInput(blob)
        age_preds = age_model.forward()
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()

        age = age_list[np.argmax(age_preds)]
        gender = gender_list[np.argmax(gender_preds)]

        # Draw rectangle around face and label with age and gender
        cv2.rectangle(frame, (x, y), (x+w, y+h), frame_color, 2)
        cv2.putText(frame, f"Age: {age}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.15, text_color, 3)
        cv2.putText(frame, f"Gender: {gender}", (x, y - -400), cv2.FONT_HERSHEY_SIMPLEX, 1.15, text_color, 3)

    return frame

# Open video capture
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_attributes(frame)
    frame = cv2.resize(frame, (resize_width, resize_height))
    cv2.imshow('Video', frame)

    # Check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
