from flask import Flask, request, jsonify,render_template
from PIL import Image
import cv2 
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import pickle
from glass_detector import glass_detection
from skin_color_detector import skin_color_detection
from gender_classification import predict_gender
from hair_beard_classification import hair_beard_detector
import json
from logging import FileHandler,WARNING
import tensorflow as tf

app = app = Flask(__name__)
file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)
@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'})

    print('1')
    
    image = request.files['image']


    pil_image = Image.open(image)


    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)


    cv2_image = cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGB)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    results = face_mesh.process(cv2_image)

    face = results.multi_face_landmarks[0].landmark

    edges , glasses_prediction = glass_detection(cv2_image,face)

    gender_prediction = predict_gender(cv2_image)

    skin_color_prediction = skin_color_detection(cv2_image,face)
    skin_color_prediction = skin_color_prediction.tolist()
    skin_color_prediction = json.dumps(skin_color_prediction)

    hair , beard  = hair_beard_detector(cv2_image)
    
    predictions = {
        'glasses': glasses_prediction,
        'skin_color_rgb':skin_color_prediction,
        'gender':gender_prediction,
        'hair_color':hair,
        'beard':beard
    }

    return jsonify(predictions)
    #return render_template('result.html', image_path='edges.jpg')

app.run(port=5000)