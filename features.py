import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image


# -------------------------
# FFT FEATURES
# -------------------------
def fft_features(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.log1p(np.abs(fshift))

    h, w = gray.shape
    cy, cx = h // 2, w // 2

    size = 20

    low = magnitude[
        max(0, cy-size):cy+size,
        max(0, cx-size):cx+size
    ]

    high = magnitude.copy()
    high[max(0, cy-size):cy+size, max(0, cx-size):cx+size] = 0

    low_energy = np.mean(low)
    high_energy = np.mean(high)

    ratio = high_energy / (low_energy + 1e-6)

    return [low_energy, high_energy, ratio]


# -------------------------
# LBP FEATURES
# -------------------------
def lbp_features(gray):
    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2),
        density=True
    )

    return hist.astype(np.float32).tolist()


# -------------------------
# GLARE FEATURES
# -------------------------
def glare_features(gray):
    threshold = 245 / 255.0
    mask = gray > threshold

    ratio = np.sum(mask) / gray.size

    if np.sum(mask) > 0:
        mean = np.mean(gray[mask])
        std = np.std(gray[mask])
    else:
        mean = 0
        std = 0

    return [ratio, mean, std]


# -------------------------
# EDGE FEATURES
# -------------------------
def edge_features(gray):
    edges = cv2.Canny((gray * 255).astype(np.uint8), 100, 200)

    density = np.sum(edges > 0) / edges.size

    pixels = gray[edges > 0]

    if len(pixels) > 0:
        mean = np.mean(pixels)
        std = np.std(pixels)
    else:
        mean = 0
        std = 0

    return [density, mean, std]


# -------------------------
# MAIN FEATURE FUNCTION
# -------------------------
def extract_features(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
    except Exception as e:
        raise ValueError(f"Cannot read image: {image_path} -> {e}")

    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = np.array(gray, dtype=np.float32)
    gray = gray / 255.0

    features = []
    features += fft_features(gray)
    features += lbp_features(gray)
    features += glare_features(gray)
    features += edge_features(gray)

    return np.array(features, dtype=np.float32)