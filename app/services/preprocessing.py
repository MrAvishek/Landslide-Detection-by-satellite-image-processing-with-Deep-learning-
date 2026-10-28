import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def normalize_rgb(image):
    """Normalize B2, B3 and B4 to the range 0–1."""
    rgb = image[:, :, [1, 2, 3]].astype(np.float32)

    min_val = rgb.min(axis=(0, 1), keepdims=True)
    max_val = rgb.max(axis=(0, 1), keepdims=True)

    return (rgb - min_val) / (max_val - min_val + 1e-6)


def calculate_indices(image):
    """Calculate NDVI, NDMI and NBR."""
    red = image[:, :, 3]
    nir = image[:, :, 7]
    swir1 = image[:, :, 10]
    swir2 = image[:, :, 11]

    eps = 1e-6

    ndvi = (nir - red) / (nir + red + eps)
    ndmi = (nir - swir1) / (nir + swir1 + eps)
    nbr = (nir - swir2) / (nir + swir2 + eps)

    return np.stack([ndvi, ndmi, nbr], axis=-1)


def calculate_grayscale(image):
    """Calculate grayscale from B2, B3 and B4."""
    blue = image[:, :, 1]
    green = image[:, :, 2]
    red = image[:, :, 3]

    grayscale = (blue + green + red) / 3.0
    return grayscale[:, :, np.newaxis]


def calculate_filters(image):
    """Create Gaussian and median filtered grayscale features."""
    grayscale = calculate_grayscale(image)[:, :, 0]

    gaussian = gaussian_filter(grayscale, sigma=1)
    median = median_filter(grayscale, size=10)

    return np.stack([gaussian, median], axis=-1)


def calculate_gradients(image):
    """Calculate horizontal and vertical grayscale gradients."""
    grayscale = calculate_grayscale(image)[:, :, 0]
    vertical_gradient, horizontal_gradient = np.gradient(grayscale)

    return np.stack(
        [horizontal_gradient, vertical_gradient],
        axis=-1
    )


def calculate_edges(image):
    """Detect Canny edges in the grayscale image."""
    grayscale = calculate_grayscale(image)[:, :, 0]

    gray_min = grayscale.min()
    gray_max = grayscale.max()

    gray_scaled = (
        (grayscale - gray_min)
        / (gray_max - gray_min + 1e-6)
        * 255
    ).astype(np.uint8)

    edges = cv2.Canny(gray_scaled, 100, 200)

    return edges[:, :, np.newaxis].astype(np.float32) / 255.0


def create_features(image):
    """
    Convert a (128, 128, 14) image into
    a (128, 128, 26) feature tensor.
    """
    image = np.asarray(image, dtype=np.float32)

    if image.shape != (128, 128, 14):
        raise ValueError(
            f"Expected image shape (128, 128, 14), got {image.shape}"
        )

    rgb = normalize_rgb(image)
    indices = calculate_indices(image)
    grayscale = calculate_grayscale(image)
    filtered = calculate_filters(image)
    gradients = calculate_gradients(image)
    edges = calculate_edges(image)

    features = np.concatenate(
        [image, rgb, indices, grayscale, filtered, gradients, edges],
        axis=-1
    )

    if features.shape != (128, 128, 26):
        raise ValueError(f"Unexpected feature shape: {features.shape}")

    return features.astype(np.float32)