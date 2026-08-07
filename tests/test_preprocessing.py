import numpy as np
import pytest
from src.utils.preprocessing import normalize_minmax, normalize_zscore, apply_clahe, histogram_equalization, remove_hot_pixels

def test_normalize_minmax():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    norm = normalize_minmax(img)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0

def test_normalize_zscore():
    img = np.random.randn(64, 64) * 50 + 100
    norm = normalize_zscore(img)
    assert np.isclose(norm.mean(), 0.0, atol=1e-1)
    assert np.isclose(norm.std(), 1.0, atol=1e-1)

def test_clahe():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    clahe_img = apply_clahe(img)
    assert clahe_img.shape == img.shape
    assert clahe_img.dtype == np.uint8

def test_histogram_equalization():
    img = np.random.randint(50, 100, (64, 64), dtype=np.uint8)
    eq_img = histogram_equalization(img)
    assert eq_img.shape == img.shape
    assert eq_img.dtype == np.uint8
    assert eq_img.std() > img.std()

def test_preprocessing_pipeline():
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    img_clahe = apply_clahe(img)
    img_norm = normalize_minmax(img_clahe)
    assert img_norm.shape == (64, 64)
    assert img_norm.min() >= 0.0
    assert img_norm.max() <= 1.0

def test_hot_pixel_removal():
    img = np.zeros((64, 64), dtype=np.float32)
    img[32, 32] = 1000.0
    cleaned = remove_hot_pixels(img, threshold=500.0)
    assert cleaned[32, 32] < 1000.0
