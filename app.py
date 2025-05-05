from __future__ import division, print_function
import os
import numpy as np
import cv2
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'model.keras'  # Use 'model.h5' if still using HDF5 format
model = load_model(MODEL_PATH)

# Image preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

# Class name mapping
def getClassName(classNo):
    classes = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    if 0 <= classNo < len(classes):
        return classes[classNo]
    return "Unknown"

# Image prediction function
def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)  # Model input shape
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)[0]
    result = getClassName(classIndex)
    return result

# Route: Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route: Prediction
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        os.makedirs(upload_path, exist_ok=True)
        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)

        # Predict
        preds = model_predict(file_path, model)
        return preds

    return None

# Run the app
if __name__ == '__main__':
    app.run(port=5001, debug=True)
