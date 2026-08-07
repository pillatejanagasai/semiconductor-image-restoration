import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

logger = logging.getLogger(__name__)

def normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize image using min-max or z-score normalization."""
    img_float = image.astype(np.float32)
    if method == 'minmax':
        img_min = np.min(img_float)
        img_max = np.max(img_float)
        if img_max > img_min:
            return (img_float - img_min) / (img_max - img_min)
        else:
            return np.zeros_like(img_float)
    elif method == 'zscore':
        mean = np.mean(img_float)
        std = np.std(img_float)
        if std > 0:
            return (img_float - mean) / std
        else:
            return np.zeros_like(img_float)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def enhance_contrast_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if image.dtype != np.uint8:
        # Assuming normalized [0, 1] input if not uint8
        img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = clahe.apply(img_uint8)
    
    if image.dtype != np.uint8:
        return result.astype(np.float32) / 255.0
    return result

def histogram_equalization(image: np.ndarray, adaptive: bool = False) -> np.ndarray:
    """Apply standard or adaptive histogram equalization."""
    if adaptive:
        return enhance_contrast_clahe(image)
        
    if image.dtype != np.uint8:
        img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image
        
    result = cv2.equalizeHist(img_uint8)
    
    if image.dtype != np.uint8:
        return result.astype(np.float32) / 255.0
    return result

def adjust_brightness(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Adjust image brightness by a factor."""
    img_float = image.astype(np.float32)
    img_adjusted = img_float * factor
    if image.dtype == np.uint8:
        return np.clip(img_adjusted, 0, 255).astype(np.uint8)
    return np.clip(img_adjusted, 0.0, 1.0)

def adjust_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Adjust image contrast by a factor."""
    img_float = image.astype(np.float32)
    mean = np.mean(img_float)
    img_adjusted = (img_float - mean) * factor + mean
    if image.dtype == np.uint8:
        return np.clip(img_adjusted, 0, 255).astype(np.uint8)
    return np.clip(img_adjusted, 0.0, 1.0)

def remove_hot_pixels(image: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Remove hot pixels using median filter comparison."""
    median_filtered = cv2.medianBlur(image, 3)
    diff = np.abs(image.astype(np.float32) - median_filtered.astype(np.float32))
    std_diff = np.std(diff)
    
    mask = diff > (threshold * std_diff)
    result = image.copy()
    result[mask] = median_filtered[mask]
    return result

def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Apply edge-preserving bilateral filter."""
    if image.dtype != np.uint8 and image.dtype != np.float32:
        img = image.astype(np.float32)
    else:
        img = image
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

class PreprocessingPipeline:
    """Orchestrates the complete preprocessing pipeline for SEM images."""
    
    def __init__(self, steps: Optional[List[Dict[str, Any]]] = None):
        self.steps = []
        if steps is None:
            self.add_step('normalize', normalize_image)
            self.add_step('clahe', enhance_contrast_clahe)
            self.add_step('hot_pixels', remove_hot_pixels)
        else:
            for step in steps:
                self.add_step(step['name'], step['func'], **step.get('kwargs', {}))
    
    def add_step(self, name: str, func: Callable, **kwargs):
        """Add a step to the pipeline."""
        self.steps.append({'name': name, 'func': func, 'kwargs': kwargs})
        logger.debug(f"Added preprocessing step: {name}")
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply all steps sequentially, log each step."""
        result = image.copy()
        for step in self.steps:
            name = step['name']
            func = step['func']
            kwargs = step['kwargs']
            logger.debug(f"Applying step: {name}")
            result = func(result, **kwargs)
        return result
    
    def process_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Process a batch of images."""
        return [self.process(img) for img in images]
