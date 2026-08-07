import os
import glob
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Callable
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile

logger = logging.getLogger(__name__)

class SEMDataset(Dataset):
    """Dataset for Scanning Electron Microscope (SEM) semiconductor images.
    
    Supports two modes:
    1. Paired mode: Loads matched degraded/clean image pairs from separate directories.
    2. Synthetic mode: Loads clean images and applies synthetic degradation on-the-fly.
    
    Supported formats: PNG, JPEG, TIFF, BMP.
    """
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    
    def __init__(self, root_dir, split='train', paired_mode=False, transform=None, degradation_transform=None, patch_size=256, formats=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.paired_mode = paired_mode
        self.transform = transform
        self.degradation_transform = degradation_transform
        self.patch_size = patch_size
        self.formats = formats if formats else list(self.SUPPORTED_FORMATS)
        
        self.split_dir = self.root_dir / self.split
        
        self.clean_images = []
        self.degraded_images = []
        
        if not self.split_dir.exists():
            # Don't error out immediately, maybe it's just initialized
            logger.warning(f"Split directory does not exist: {self.split_dir}")
            return
            
        if self.paired_mode:
            clean_dir = self.split_dir / 'clean'
            degraded_dir = self.split_dir / 'degraded'
            
            if not clean_dir.exists() or not degraded_dir.exists():
                logger.warning(f"Paired mode requires 'clean' and 'degraded' subdirectories in {self.split_dir}")
                return
                
            clean_files = self._scan_images(clean_dir, self.formats)
            
            for c_path in clean_files:
                basename = c_path.name
                d_path = degraded_dir / basename
                if d_path.exists():
                    self.clean_images.append(c_path)
                    self.degraded_images.append(d_path)
                else:
                    logger.warning(f"Degraded pair not found for {c_path}")
                    
        else:
            clean_dir = self.split_dir / 'clean'
            if not clean_dir.exists():
                clean_dir = self.split_dir
            
            self.clean_images = self._scan_images(clean_dir, self.formats)
            
        if len(self.clean_images) == 0:
            logger.warning(f"No images found in {self.split_dir} matching formats {self.formats}")
        else:
            logger.info(f"Loaded {len(self.clean_images)} images from {self.split_dir}")

    def _scan_images(self, directory, formats):
        files = []
        for ext in formats:
            ext = ext.lower()
            files.extend(directory.rglob(f"*{ext}"))
            files.extend(directory.rglob(f"*{ext.upper()}"))
        return sorted(list(set(files)))
    
    def _load_image(self, path):
        ext = path.suffix.lower()
        if ext in {'.tif', '.tiff'}:
            img = tifffile.imread(str(path))
        else:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
            
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32)
            
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        
        return img
    
    def _random_crop(self, images, patch_size):
        if not images:
            return images
            
        h, w = images[0].shape[:2]
        if h < patch_size or w < patch_size:
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            padded_images = []
            for img in images:
                padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
                padded_images.append(padded)
            images = padded_images
            h, w = images[0].shape[:2]
            
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        
        cropped_images = []
        for img in images:
            crop = img[top:top + patch_size, left:left + patch_size]
            cropped_images.append(crop)
            
        return cropped_images
    
    def __len__(self):
        return len(self.clean_images)
    
    def __getitem__(self, idx):
        clean_path = self.clean_images[idx]
        clean_img = self._load_image(clean_path)
        
        if self.paired_mode:
            degraded_path = self.degraded_images[idx]
            degraded_img = self._load_image(degraded_path)
            
            if self.patch_size > 0:
                clean_img, degraded_img = self._random_crop([clean_img, degraded_img], self.patch_size)
        else:
            if self.patch_size > 0:
                clean_img = self._random_crop([clean_img], self.patch_size)[0]
                
            degraded_img = clean_img.copy()
            if self.degradation_transform:
                res = self.degradation_transform(image=degraded_img)
                degraded_img = res['image']
                
        if self.transform:
            if self.paired_mode:
                res = self.transform(image=clean_img, image0=degraded_img)
                clean_img = res['image']
                degraded_img = res['image0']
            else:
                res = self.transform(image=clean_img)
                clean_img = res['image']
                
        if len(clean_img.shape) == 2:
            clean_tensor = torch.from_numpy(clean_img).unsqueeze(0)
            degraded_tensor = torch.from_numpy(degraded_img).unsqueeze(0)
        else:
            clean_tensor = torch.from_numpy(clean_img).permute(2, 0, 1)
            degraded_tensor = torch.from_numpy(degraded_img).permute(2, 0, 1)
            
        metadata = {
            'filename': clean_path.name,
            'original_shape': clean_img.shape,
            'paired': self.paired_mode
        }
        
        return {
            'degraded': degraded_tensor,
            'clean': clean_tensor,
            'filename': clean_path.name,
            'metadata': metadata
        }
