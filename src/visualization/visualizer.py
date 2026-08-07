import os
import logging
from typing import Optional, List, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

logger = logging.getLogger(__name__)

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array for visualization."""
    if tensor.dim() == 4:
        tensor = tensor[0] # Take first from batch
    tensor = tensor.squeeze(0).cpu().detach().clamp(0, 1)
    return tensor.numpy()

def create_comparison(degraded: np.ndarray, restored: np.ndarray, clean: Optional[np.ndarray] = None, title: str = '') -> plt.Figure:
    """Create side-by-side before/after comparison."""
    n_cols = 3 if clean is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), dpi=150)
    
    if n_cols == 2:
        axes = [axes[0], axes[1]]
    
    axes[0].imshow(degraded, cmap='gray')
    axes[0].set_title('Degraded')
    axes[0].axis('off')
    
    axes[1].imshow(restored, cmap='gray')
    axes[1].set_title('Restored')
    axes[1].axis('off')
    
    if clean is not None:
        axes[2].imshow(clean, cmap='gray')
        axes[2].set_title('Clean (Ground Truth)')
        axes[2].axis('off')
        
    if title:
        fig.suptitle(title, fontsize=16)
        
    fig.tight_layout()
    return fig

def create_difference_map(image1: np.ndarray, image2: np.ndarray, title: str = 'Difference Map') -> plt.Figure:
    """Create absolute difference heatmap between two images."""
    diff = np.abs(image1 - image2)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    
    im = ax.imshow(diff, cmap='hot')
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax)
    
    fig.tight_layout()
    return fig

def create_edge_map(image: np.ndarray, method: str = 'canny', title: str = 'Edge Map') -> plt.Figure:
    """Create edge detection visualization."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    
    img_uint8 = (image * 255).astype(np.uint8)
    
    if method == 'canny':
        edges = cv2.Canny(img_uint8, 100, 200)
    elif method == 'sobel':
        sobelx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
    elif method == 'laplacian':
        edges = cv2.Laplacian(img_uint8, cv2.CV_64F)
        edges = np.abs(edges)
    else:
        edges = img_uint8
        
    ax.imshow(edges, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    
    fig.tight_layout()
    return fig

def create_heatmap(data: np.ndarray, title: str = 'Heatmap', cmap: str = 'jet') -> plt.Figure:
    """Create heatmap overlay."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax)
    
    fig.tight_layout()
    return fig

def create_metrics_chart(metrics: dict, title: str = 'Quality Metrics') -> plt.Figure:
    """Create bar chart of quality metrics."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    
    keys = list(metrics.keys())
    values = list(metrics.values())
    
    bars = ax.bar(keys, values, color='skyblue')
    ax.set_title(title)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom')
        
    fig.tight_layout()
    return fig

class VisualizationSuite:
    """Complete visualization suite for restoration results."""
    
    def __init__(self, output_dir: str = 'outputs/visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_result(self, degraded, restored, clean=None, filename='result', save=True):
        """Generate all visualizations for a single result."""
        
        deg_np = tensor_to_numpy(degraded) if isinstance(degraded, torch.Tensor) else degraded
        res_np = tensor_to_numpy(restored) if isinstance(restored, torch.Tensor) else restored
        cln_np = tensor_to_numpy(clean) if isinstance(clean, torch.Tensor) and clean is not None else clean
        
        figs = {}
        
        figs['comparison'] = create_comparison(deg_np, res_np, cln_np)
        figs['edge_map_restored'] = create_edge_map(res_np, title='Restored Edges')
        
        if cln_np is not None:
            figs['diff_map'] = create_difference_map(res_np, cln_np, title='Restored vs Clean')
            
        if save:
            for k, fig in figs.items():
                self.save_figure(fig, f'{filename}_{k}')
                
        return figs
    
    def visualize_batch(self, degraded_batch, restored_batch, clean_batch=None, prefix='batch'):
        """Visualize a batch of results."""
        batch_size = degraded_batch.shape[0] if isinstance(degraded_batch, torch.Tensor) else len(degraded_batch)
        
        for i in range(batch_size):
            d = degraded_batch[i:i+1] if isinstance(degraded_batch, torch.Tensor) else degraded_batch[i]
            r = restored_batch[i:i+1] if isinstance(restored_batch, torch.Tensor) else restored_batch[i]
            c = clean_batch[i:i+1] if clean_batch is not None and isinstance(clean_batch, torch.Tensor) else (clean_batch[i] if clean_batch is not None else None)
            
            self.visualize_result(d, r, c, filename=f'{prefix}_{i}', save=True)
    
    def save_figure(self, fig, filename):
        """Save figure to output directory."""
        fig.savefig(self.output_dir / f'{filename}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
