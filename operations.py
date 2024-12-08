import numpy as np


def add(image1, image2):
    height, width = image1.shape
    added_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            added_value = int(image1[i, j]) + int(image2[i, j])
            added_image[i, j] = max(0, min(added_value, 255))

    return added_image


def sub(image1, image2):
    height, width = image1.shape
    sub_image = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            subtracted_value = int(image1[i, j]) - int(image2[i, j])
            sub_image[i, j] = max(0, min(subtracted_value, 255))

    return sub_image


def invert(image):
    inverted_image = 255 - image
    return inverted_image
