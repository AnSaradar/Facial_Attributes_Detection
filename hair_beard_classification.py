import cv2
import numpy as np
import tensorflow as tf 

model = tf.keras.models.load_model('model_vggface_loss_0.343691349029541.h5')

def get_labels(pred):
    hair_indices = [8,9,11,17]
    hair = {8:"Black_Hair",9:"Blond_Hair",11:"Brown_Hair",17:"Gray_Hair"}
    max_hair_index = hair_indices[0]
    max_value = pred[max_hair_index]

    for index in hair_indices:
        if pred[index] > max_value:
            max_value = pred[index]
            max_hair_index = index

    beard = ''
    if pred[24]>0.75:
        beard = 'No Beard'
    else:
         beard = 'Beard'

    features = {'Hair':hair[max_hair_index],'Beard':beard}
    
    return features


def hair_beard_detector(image):
    
    image = image.astype(float) / 255.0
    image = cv2.resize(image, (112, 112))
    converted_image = image[np.newaxis, ...]
    pred_labels = model.predict(converted_image)
    reshaped_array = pred_labels.reshape(40)
    return get_labels(reshaped_array)