from PIL import Image
import cv2 
import numpy as np
import matplotlib.pyplot as plt

def glass_detection(image, face):
    nose_bridge_x = []
    nose_bridge_y = []
    for i in  [9,8,245,465]:
        x = int(face[i].x * image.shape[1])
        y = int(face[i].y * image.shape[0])
        nose_bridge_x.append(x)
        nose_bridge_y.append(y)

    ## x_min and x_max
    x_min = min(nose_bridge_x)
    x_max = max(nose_bridge_x)
    ### ymin (from top eyebrow coordinate),  ymax
    y_min = min(nose_bridge_y)
    y_max = max(nose_bridge_y)

    nose_image = image[y_min:y_max, x_min:x_max]

    img_blur = cv2.GaussianBlur(np.array(nose_image),(3,3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)
    

    edges_center = edges.T[(int(len(edges.T)/2))]


    if 255 in edges_center:
        return edges , True
    else:
        return edges , False