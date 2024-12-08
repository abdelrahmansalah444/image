import os
os.system('pip install opencv-python')
from scipy.signal import find_peaks
import cv2
import numpy as np


def manual_segmentation(image, low_T, high_T, value=255):
    segmented_image_manual = np.zeros_like(image)
    segmented_image_manual[(image >= low_T) & (image <= high_T)] = value
    return segmented_image_manual


def histogram_peak_segmentation(image):

    histogram = np.zeros(256, dtype=int)
    for row in image:
        for pixel in row:
            histogram[pixel] += 1

    peaks_indices = find_histogram_peaks(histogram)
    low_T, high_T = calculate_thresholds(peaks_indices, histogram)

    segmented_image_peak = np.zeros_like(image)
    segmented_image_peak[(image >= low_T) & (image <= high_T)] = 255

    return segmented_image_peak


def find_histogram_peaks(histogram):
    peaks, _ = find_peaks(histogram, height=0)
    sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
    return sorted_peaks[:2]


def calculate_thresholds(peaks_indices, histogram):
    peak1 = peaks_indices[0]
    peak2 = peaks_indices[1]
    low_T = (peak1 + peak2)//2
    high_T = peak2

    return low_T, high_T


def histogram_valley_segmentation(image):

    histogram = np.zeros(256, dtype=int)
    for row in image:
        for pixel in row:
            histogram[pixel] += 1

    peaks_indices = find_histogram_peaks(histogram)
    valley_point = find_valley_point(peaks_indices, histogram)
    low_T, high_T = valley_high_low(peaks_indices, valley_point)

    segmented_image_vally = np.zeros_like(image)
    segmented_image_vally[(image >= low_T) & (image <= high_T)] = 255
    return segmented_image_vally


def find_histogram_peaks(histogram):
    peaks, _ = find_peaks(histogram, height=0)
    sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
    return sorted_peaks[:2]


def find_valley_point(peaks_indices, histogram):
    valley_point = 0
    min_valley = float('inf')
    start, end = peaks_indices
    for i in range(start, end + 1):
        if histogram[i] < min_valley:
            min_valley = histogram[i]
            valley_point = i
    return valley_point


def valley_high_low(peaks_indices, valley_point):
    low_T = valley_point
    high_T = peaks_indices[1]
    return low_T, high_T


def adaptive_histogram_segmentation(image):

    histogram = np.zeros(256, dtype=int)
    for row in image:
        for pixel in row:
            histogram[pixel] += 1

    peaks_indices = find_histogram_peaks(histogram)
    low_T, high_T = valley_high_low(
        peaks_indices, find_valley_point(peaks_indices, histogram))

    segmented_image = np.zeros_like(image)
    segmented_image[(image >= low_T) & (image <= high_T)] = 255
    background_mean, object_mean = calculate_means(segmented_image, image)
    new_peaks_indices = [int(background_mean), int(object_mean)]
    new_low_T, new_high_T = valley_high_low(new_peaks_indices, find_valley_point(
        new_peaks_indices, cv2.calcHist([image], [0], None, [256], [0, 255]).flatten()))

    final_segmented_image = np.zeros_like(image)
    final_segmented_image[(image >= new_low_T) & (image <= new_high_T)] = 255
    return final_segmented_image


def find_histogram_peaks(histogram):
    peaks, _ = find_peaks(histogram, height=0)
    sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
    return sorted_peaks[:2]


def find_valley_point(peaks_indices, histogram):
    valley_point = 0
    min_valley = float('inf')
    start, end = peaks_indices
    for i in range(start, end + 1):
        if histogram[i] < min_valley:
            min_valley = histogram[i]
            valley_point = i
    return valley_point


def valley_high_low(peaks_indices, valley_point):
    low_T = valley_point
    high_T = peaks_indices[1]
    return low_T, high_T


def calculate_means(segmented_image, original_image):
    object_sum = 0
    object_count = 0
    background_sum = 0
    background_count = 0

    for i in range(segmented_image.shape[0]):
        for j in range(segmented_image.shape[1]):
            if segmented_image[i, j] == 255:  # Object pixel
                object_sum += original_image[i, j]
                object_count += 1
            else:  # Background pixel
                background_sum += original_image[i, j]
                background_count += 1

    object_mean = object_sum / object_count if object_count > 0 else 0
    background_mean = background_sum / background_count if background_count > 0 else 0

    return background_mean, object_mean