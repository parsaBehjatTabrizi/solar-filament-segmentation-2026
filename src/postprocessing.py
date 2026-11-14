"""Probability map -> filament instances (threshold, morphological closing, connected components, size filter)."""
import cv2
import numpy as np


def extract_instances(prob: np.ndarray, threshold: float = 0.5, closing_radius: int = 10, min_pixels: int = 400):
    """Return a list of boolean instance masks.

    prob            float probability map at the original image resolution (2048 x 2048)
    threshold       pixels with prob > threshold are foreground
    closing_radius  radius (px) of the elliptical structuring element that re-joins fragments (0 disables the closing)
    min_pixels      connected components with an area <= min_pixels are discarded
    """
    binary = (prob > threshold).astype(np.uint8)
    if closing_radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * closing_radius + 1, 2 * closing_radius + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    areas = np.bincount(labels.ravel(), minlength=n)
    return [labels == k for k in range(1, n) if areas[k] > min_pixels]
