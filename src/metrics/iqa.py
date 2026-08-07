import cv2
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def estimate_noise(image: np.ndarray) -> float:
    """Estimate noise level using the Laplacian method (Immerkaer's method)."""
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
        
    img_float = img_gray.astype(np.float64)
    H, W = img_float.shape
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    sigma = np.sum(np.abs(cv2.filter2D(img_float, -1, M)))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))
    return float(sigma)

def estimate_blur(image: np.ndarray) -> float:
    """Estimate blur level using variance of Laplacian."""
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(laplacian.var())

def estimate_sharpness(image: np.ndarray) -> float:
    """Estimate sharpness using gradient magnitude (Sobel)."""
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return float(np.mean(magnitude))

def estimate_contrast(image: np.ndarray) -> float:
    """Estimate contrast using Michelson contrast."""
    img_float = image.astype(np.float64)
    max_val = np.max(img_float)
    min_val = np.min(img_float)
    epsilon = 1e-6
    return float((max_val - min_val) / (max_val + min_val + epsilon))

def estimate_brightness(image: np.ndarray) -> float:
    """Estimate brightness as mean luminance."""
    return float(np.mean(image))

def compute_snr(image: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio."""
    img_float = image.astype(np.float64)
    mean_val = np.mean(img_float)
    std_val = np.std(img_float)
    if std_val == 0:
        return float('inf')
    return float(mean_val / std_val)

def compute_entropy(image: np.ndarray) -> float:
    """Compute image entropy (information content)."""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.astype(np.uint8)
    else:
        img = image
        
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    hist = hist[hist > 0]
    p = hist / np.sum(hist)
    entropy = -np.sum(p * np.log2(p))
    return float(entropy)

class ImageQualityAssessor:
    """Comprehensive image quality assessment for SEM images."""
    
    def __init__(self):
        self.metrics = {
            'noise': estimate_noise,
            'blur_laplacian_var': estimate_blur,
            'sharpness_gradient': estimate_sharpness,
            'contrast_michelson': estimate_contrast,
            'brightness_mean': estimate_brightness,
            'snr': compute_snr,
            'entropy': compute_entropy
        }
    
    def assess(self, image: np.ndarray) -> Dict[str, float]:
        """Run all quality metrics on an image."""
        results = {}
        for name, func in self.metrics.items():
            results[name] = func(image)
        return results
    
    def compare(self, degraded: np.ndarray, restored: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compare quality metrics between degraded and restored images."""
        degraded_metrics = self.assess(degraded)
        restored_metrics = self.assess(restored)
        
        comparison = {}
        for metric in self.metrics.keys():
            comparison[metric] = {
                'degraded': degraded_metrics[metric],
                'restored': restored_metrics[metric],
                'delta': restored_metrics[metric] - degraded_metrics[metric]
            }
        return comparison
    
    def generate_report(self, results: Dict) -> str:
        """Generate a formatted quality assessment report."""
        report = "Image Quality Assessment Report\n"
        report += "=" * 30 + "\n"
        for metric, values in results.items():
            report += f"{metric}:\n"
            report += f"  Degraded: {values['degraded']:.4f}\n"
            report += f"  Restored: {values['restored']:.4f}\n"
            report += f"  Delta:    {values['delta']:+.4f}\n"
            report += "-" * 30 + "\n"
        return report
