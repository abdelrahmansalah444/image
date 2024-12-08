import os
os.system('pip install Pillow')
from PIL import Image
import numpy as np


def Halftone(gray_image):
    gray_image = gray_image.copy()
    h, w = gray_image.shape
    new_image = np.zeros((h, w), dtype=np.uint8)

    for row in range(h):   # row
        for col in range(w):    # column
            new_image[row][col] = 255 if (
                gray_image[row][col] < 128) else 0    # threshold
            error = gray_image[row][col] - new_image[row][col]

            if (col > 0) and (row < (h-1)):
                gray_image[row+1][col-1] += (3/16) * error
            if row < (h-1):
                gray_image[row+1][col] += (5/16) * error
            if (col < (w-1)):
                gray_image[row][col+1] += (7/16) * error
            if (col < (w-1)) and (row < (h-1)):
                gray_image[row+1][col+1] += (1/16) * error

    return new_image  # halftone_image


def Simple_halftone(gray_image, threshold):
    # Apply the threshold
    gray_image = gray_image.copy()
    h, w = gray_image.shape
    new_image = np.zeros((h, w), dtype=np.uint8)

    for row in range(h):   # row
        for col in range(w):    # column
            new_image[row][col] = 255 if (
                gray_image[row][col] < threshold) else 0    # threshold

    return new_image


def Histogram_equalization(gray_image):
    gray_image = gray_image.copy()
    h, w = gray_image.shape
    new_image = np.zeros((h, w), dtype=np.uint8)

    # occurance of each pixel value (histogram)
    gray_levels_list = [0] * 256
    for row in range(h):   # row
        for col in range(w):    # column
            pixel_value = gray_image[row][col]
            gray_levels_list[pixel_value] += 1

    cdf_list = [0] * 256
    pdf_list = [0]*256
    sum = 0
    hist_level_list = [0]*256

    for i in range(256):
        # calculate PDF
        pdf_value = gray_levels_list[i] / (w*h)
        pdf_list[i] = pdf_value

        # calculate CDF
        sum += pdf_list[i]
        cdf_list[i] = sum if sum < 256 else 256

        # calculate hist level
        hist_level_list[i] = round(cdf_list[i] * 255)

    for row in range(h):   # row
        for col in range(w):
            old_pixel_value = gray_image[row][col]
            hist_level_val = hist_level_list[old_pixel_value]
            new_image[row][col] = hist_level_val

    h, w = new_image.shape
    histogram_equalization_list = [0] * 256
    for row in range(h):   # row
        for col in range(w):    # column
            pixel_value = new_image[row][col]
            histogram_equalization_list[pixel_value] += 1

    return new_image, hist_level_list, histogram_equalization_list


def gray_scale(image):
    image_array = np.array(image)
    if len(image_array.shape) == 3:  # RGB
        # grayscale using luminosity formula: 0.2989*R + 0.5870*G + 0.1140*B
        grayscale_array = (
            # img[height , width , channel(red,green,blue)]
            0.2989 * image_array[:, :, 0] +
            0.5870 * image_array[:, :, 1] +
            0.1140 * image_array[:, :, 2]
        ).astype(np.uint8)

        # Convert back to an image
        return Image.fromarray(grayscale_array)
    else:
        return Image.fromarray(image_array)  # grayscale already


def calculate_threshold(image):
    image_array = np.array(image)

    threshold = np.mean(image_array)  # mean of all pixels

    binary_image = image_array >= threshold   # each pixel false or true
    binary_image = binary_image.astype(np.uint8)  # convert true ,false to 1,0
    binary_image = binary_image * 255   # 0 and 255

    return threshold, Image.fromarray(binary_image)
