import cv2
import numpy as np

image = cv2.imread('example.jpg', 0)
maxValue = np.amax(image)

print maxValue