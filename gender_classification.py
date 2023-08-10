import cv2
import numpy as np
import tensorflow as tf 


# Example for a TensorFlow/Keras model
model = tf.keras.models.load_model('face_gender_model.h5')

def preprocess_image(image):
    #input_image = cv2.imread(image_path)
    input_image = cv2.resize(image, (150, 150))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGRA2BGR)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image / 255.0  # Normalize the image
    return input_image



def predict_gender(image):
    test_image = preprocess_image(image)
    prediction = model.predict(test_image)
    if prediction[0] < 0.5:
        gender = 'Female'
    else:
        gender = 'Male'

    return gender
