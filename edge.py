import os
os.system('pip install opencv-python')
import cv2
import numpy as np


MASK_7X7 = np.array([[0, 0, -1, -1, -1, 0, 0],
                     [0, -2, -3, -3, -3, -2, 0],
                     [-1, -3, 5, 5, 5, -3, -1],
                     [-1, -3, 5, 16, 5, -3, -1],
                     [-1, -3, 5, 5, 5, -3, -1],
                     [0, -2, -3, -3, -3, -2, 0],
                     [0, 0, -1, -1, -1, 0, 0]], dtype=np.float32)

MASK_9X9 = np.array([[0, 0, 0, -1, -1, -1, 0, 0, 0],
                     [0, -2, -3, -3, -3, -3, -3, -2, 0],
                     [0, -3, -2, -1, -1, -1, -2, -3, 0],
                     [-1, -3, -1, 9, 9, 9, -1, -3, -1],
                     [-1, -3, -1, 9, 19, 9, -1, -3, -1],
                     [-1, -3, -1, 9, 9, 9, -1, -3, -1],
                     [0, -3, -2, -1, -1, -1, -2, -3, 0],
                     [0, -2, -3, -3, -3, -3, -3, -2, 0],
                     [0, 0, 0, -1, -1, -1, 0, 0, 0]], dtype=np.float32)


def sobel_edge_detection(image, treshold=10):

    MASK_SOBEL = {'GX': np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                  [-1, 0, 1]], dtype=np.float32),

                  'GY': np.array([[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]], dtype=np.float32)
                  }

    # Output images (initialize black images)
    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)
    gradient_magnitude = np.zeros_like(image, dtype=np.float32)

    # Apply Sobel filter using cv2.filter2D
    gradient_x_filtered = cv2.filter2D(image, -1, MASK_SOBEL['GX'])
    gradient_y_filtered = cv2.filter2D(image, -1, MASK_SOBEL['GY'])

   # Apply nested loops to store edges in the black images
    height, width = image.shape
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = gradient_x_filtered[i, j]
            gy = gradient_y_filtered[i, j]

            gradient_x[i, j] = gx
            gradient_y[i, j] = gy
            gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)

    # Normalize and convert to uint8
    gradient_magnitude = np.clip(
        gradient_magnitude / gradient_magnitude.max() * 255, 0, 255).astype(np.uint8)

    # Apply treshold
    gradient_magnitude = np.where(
        gradient_magnitude >= treshold, gradient_magnitude, 0)

    return gradient_magnitude


def prewitt_edge_detection(image, treshold=10):

    MASK_PREWITT = {'GX': np.array([[-1, 0, 1],
                                   [-1, 0, 1],
                                    [-1, 0, 1]], dtype=np.float32),

                    'GY': np.array([[-1, -1, -1],
                                    [0, 0, 0],
                                    [1, 1, 1]], dtype=np.float32)
                    }

    # Output images (initialize black images)
    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)
    gradient_magnitude = np.zeros_like(image, dtype=np.float32)

    # Apply Sobel filter using cv2.filter2D
    gradient_x_filtered = cv2.filter2D(image, -1, MASK_PREWITT['GX'])
    gradient_y_filtered = cv2.filter2D(image, -1, MASK_PREWITT['GY'])

   # Apply nested loops to store edges in the black images
    height, width = image.shape
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = gradient_x_filtered[i, j]
            gy = gradient_y_filtered[i, j]

            gradient_x[i, j] = gx
            gradient_y[i, j] = gy
            gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)

    # Normalize and convert to uint8
    gradient_magnitude = np.clip(
        gradient_magnitude / gradient_magnitude.max() * 255, 0, 255).astype(np.uint8)

    # Apply treshold
    gradient_magnitude = np.where(
        gradient_magnitude >= treshold, gradient_magnitude, 0)

    return gradient_magnitude


def kirsch_edge_detection(image):
    MASK_KIRSCH = {
        'N': np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        'NW': np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        'W': np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        'SW': np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        'S': np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        'SE': np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
        'E': np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        'NE': np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
    }

    filters = {key: cv2.filter2D(image, cv2.CV_32F, kernel)
               for key, kernel in MASK_KIRSCH.items()}

    max_direction = np.max(list(filters.values()), axis=0)

    max_direction = cv2.normalize(
        max_direction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    direction = max(filters, key=lambda k: np.max(filters[k]))

    return max_direction, direction


def homogeneity_edge_detection(image, threshold=5):
    height, width = image.shape
    homogeneity_edge = np.zeros_like(image)
    image = image.astype(np.float32)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            central_pixel = image[i, j]
            differeneces = [np.abs(central_pixel - image[i-1, j-1]),
                            np.abs(central_pixel - image[i-1, j]),
                            np.abs(central_pixel - image[i-1, j+1]),
                            np.abs(central_pixel - image[i, j-1]),
                            np.abs(central_pixel - image[i, j+1]),
                            np.abs(central_pixel - image[i+1, j-1]),
                            np.abs(central_pixel - image[i+1, j]),
                            np.abs(central_pixel - image[i+1, j+1])]

            homogeneity_max = np.max(differeneces)
            homogeneity_edge[i, j] = homogeneity_max
            homogeneity_edge[i, j] = np.where(
                homogeneity_edge[i, j] >= threshold, homogeneity_edge[i, j], 0)

    return homogeneity_edge.astype(np.uint8)


def difference_edge_detection(image, threshold=10):
    height, width = image.shape
    difference_edge = np.zeros_like(image)
    image = image.astype(np.float32)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            diff1 = np.abs(image[i-1, j-1]-image[i+1, j+1])
            diff2 = np.abs(image[i-1, j+1]-image[i+1, j-1])
            diff3 = np.abs(image[i, j-1]-image[i, j+1])
            diff4 = np.abs(image[i-1, j]-image[i+1, j])

            max_diff = np.max([diff1, diff2, diff3, diff4])
            difference_edge[i, j] = max_diff
            difference_edge[i, j] = np.where(
                difference_edge[i, j] > threshold, difference_edge[i, j], 0)

    return difference_edge.astype(np.uint8)


def difference_of_gaussians_edge_detection(image):
    blured_image1 = cv2.filter2D(image, -1, MASK_7X7)
    blured_image2 = cv2.filter2D(image, -1, MASK_9X9)

    difference_of_gaussians = blured_image1-blured_image2
    return difference_of_gaussians, blured_image1, blured_image2


def contrast_edge_detection(image):
    CONTRAST_MASK = np.array([[-1, 0, -1],
                              [0, 4, 0],
                              [-1, 0, -1]])

    SMOTHING_MASK = 1/9 * np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])
    edge_output = cv2.filter2D(image, -1, CONTRAST_MASK)
    smoothed_image = cv2.filter2D(image, -1, SMOTHING_MASK)
    smoothed_image = smoothed_image.astype(np.float32)
    smoothed_image += 1e-20
    contrast_edge = edge_output / smoothed_image
    contrast_edge_normalized = cv2.normalize(contrast_edge, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return np.clip(contrast_edge_normalized, 0, 255).astype(np.uint8)    


def variance_edge_detection(image):
    height, width = image.shape
    variance_edge = np.zeros_like(image)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            mean = np.mean(neighborhood)
            variance_value = np.sum((neighborhood - mean) ** 2) / 9
            variance_edge[i, j] = variance_value
    return variance_edge


def range_edge_detection(image):
    height, width = image.shape
    range_edge = np.zeros_like(image)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            range_value = np.max(neighborhood) - np.min(neighborhood)
            range_edge[i, j] = range_value
    return range_edge
