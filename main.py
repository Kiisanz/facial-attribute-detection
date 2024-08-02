instafrom flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

# Load pre-trained models for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load models for age and gender prediction
age_model = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
gender_model = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

# Define the list of age and gender labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

def detect_attributes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]

        # Prepare blob for age and gender
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0))
        age_model.setInput(blob)
        age_preds = age_model.forward()
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()

        age = age_list[np.argmax(age_preds)]
        gender = gender_list[np.argmax(gender_preds)]

        # Draw rectangle around face and add labels
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, f"Age: {age}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        cv2.putText(image, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        results.append({
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'age': age,
            'gender': gender
        })

    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return results, img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read image
        img_array = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Detect attributes
        results, img_base64 = detect_attributes(img)

        return jsonify({'results': results, 'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
