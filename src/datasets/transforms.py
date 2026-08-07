import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2

class PoissonNoiseTransform(ImageOnlyTransform):
    """Custom Albumentations transform to apply Poisson noise to an image."""
    
    def __init__(self, scale_range=(10.0, 50.0), always_apply=False, p=0.5):
        super(PoissonNoiseTransform, self).__init__(always_apply, p)
        self.scale_range = scale_range
        
    def apply(self, img, **params):
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        noisy = np.random.poisson(img * scale) / scale
        noisy = np.clip(noisy, 0.0, 1.0)
        return noisy.astype(np.float32)
        
    def get_transform_init_args_names(self):
        return ("scale_range",)

class SpeckleNoiseTransform(ImageOnlyTransform):
    """Speckle noise is a multiplicative noise: I = J + J * N"""
    def __init__(self, mean=0, std=(0.1, 0.4), always_apply=False, p=0.5):
        super(SpeckleNoiseTransform, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std
        
    def apply(self, img, **params):
        import random
        std = random.uniform(self.std[0], self.std[1])
        noise = np.random.normal(self.mean, std, img.shape)
        
        if img.dtype == np.uint8:
            noisy_img = img.astype(np.float32)
            noisy_img = noisy_img + noisy_img * noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        else:
            noisy_img = img + img * noise
            noisy_img = np.clip(noisy_img, 0.0, 1.0).astype(np.float32)
        return noisy_img
        
    def get_transform_init_args_names(self):
        return ("mean", "std")


def get_training_transforms(patch_size=256):
    """Returns A.Compose with standard training augmentations."""
    return A.Compose([
        A.RandomCrop(width=patch_size, height=patch_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], additional_targets={'image0': 'image'})

def get_validation_transforms(patch_size=256):
    """Returns A.Compose with center crop for validation."""
    return A.Compose([
        A.CenterCrop(width=patch_size, height=patch_size),
    ], additional_targets={'image0': 'image'})

def get_degradation_transforms():
    """Returns A.Compose simulating SEM degradation."""
    return A.Compose([
        A.GaussNoise(var_limit=(0.001, 0.01), mean=0, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        PoissonNoiseTransform(scale_range=(10.0, 50.0), p=0.5),
        SpeckleNoiseTransform(std=(0.1, 0.3), p=0.6)
    ])

def get_normalization():
    """Returns normalization transform."""
    return A.Compose([])

def apply_transforms(image, transform):
    """Applies an albumentations transform and returns the result."""
    if transform:
        res = transform(image=image)
        return res['image']
    return image
