import os
os.system('pip install opencv-python')
import cv2
import numpy as np

mask_hi_pass = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

mask_low_pass = np.array([
    [0, 1/6, 0],
    [1/6, 2/6, 1/6],
    [0, 1/6, 0]
], dtype=np.float32)


def lowpass(image):
    result = cv2.filter2D(image, -1, mask_low_pass)
    return result


def highpass(image):
    result = cv2.filter2D(image, -1, mask_hi_pass)
    return result


def median_filter(image):
    height, width = image.shape
    filtered_image = np.zeros_like(image)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            median_value = np.median(neighborhood)
            filtered_image[i, j] = median_value

    return filtered_image
