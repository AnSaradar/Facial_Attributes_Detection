import cv2 
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def skin_color_detection(image,face):
    skin_bridge_x = []
    skin_bridge_y = []


    for i in  [116,117,147,187]:
        x = int(face[i].x * image.shape[1])
        y = int(face[i].y * image.shape[0])
        skin_bridge_x.append(x)
        skin_bridge_y.append(y)

    ### x_min and x_max
    x_min = min(skin_bridge_x)
    x_max = max(skin_bridge_x)
    ### ymin (from top eyebrow coordinate),  ymax
    y_min = min(skin_bridge_y)
    y_max = max(skin_bridge_y)

    skin_image = image[y_min:y_max, x_min:x_max]

    # image_ycrcb = cv2.cvtColor(skin_image, cv2.COLOR_BGR2YCrCb)

    # lower_skin = np.array([0, 133, 77], dtype="uint8")
    # upper_skin = np.array([255, 173, 127], dtype="uint8")
    # skin_mask = cv2.inRange(image_ycrcb, lower_skin, upper_skin)
    # skin_pixels = cv2.bitwise_and(skin_image, skin_image, mask=skin_mask)
    # pixels = skin_pixels.reshape(-1, 3)
    # kmeans = KMeans(n_clusters=3)  # You can adjust the number of clusters as per your requirement
    # kmeans.fit(pixels)
    # dominant_colors = kmeans.cluster_centers_
    
    average_color = np.mean(skin_image, axis=(0, 1))

    # Convert the average color to RGB format
    average_color_rgb = np.uint8(average_color)[::-1]

    # Display the average RGB color
    print("Average Color (RGB):", average_color_rgb)



    cv2.imwrite('skin.jpg',skin_image)

    # dominant_colors_rgb = np.uint8(dominant_colors)
    # dominant_colors_rgb = dominant_colors_rgb[:, ::-1]
    return average_color_rgb