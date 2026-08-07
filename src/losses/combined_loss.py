import torch
import torch.nn as nn
from .basic_losses import L1Loss, SSIMLoss, PerceptualLoss
from .advanced_losses import EdgeLoss, FrequencyLoss, GradientLoss

class CombinedLoss(nn.Module):
    """Weighted combination of all loss functions."""
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            self.weights = {
                'l1': 1.0, 
                'ssim': 0.5, 
                'perceptual': 0.1, 
                'edge': 0.3, 
                'frequency': 0.2, 
                'gradient': 0.2
            }
        else:
            self.weights = weights
            
        self.losses = nn.ModuleDict({
            'l1': L1Loss() if self.weights.get('l1', 0) > 0 else None,
            'ssim': SSIMLoss() if self.weights.get('ssim', 0) > 0 else None,
            'perceptual': PerceptualLoss() if self.weights.get('perceptual', 0) > 0 else None,
            'edge': EdgeLoss() if self.weights.get('edge', 0) > 0 else None,
            'frequency': FrequencyLoss() if self.weights.get('frequency', 0) > 0 else None,
            'gradient': GradientLoss() if self.weights.get('gradient', 0) > 0 else None
        })
        
    def forward(self, pred, target, return_components=False):
        total_loss = 0.0
        components = {}
        
        for name, loss_fn in self.losses.items():
            if loss_fn is not None and self.weights.get(name, 0) > 0:
                loss_val = loss_fn(pred, target)
                weighted_loss = self.weights[name] * loss_val
                total_loss += weighted_loss
                components[name] = weighted_loss.item()
                
        if return_components:
            return total_loss, components
        return total_loss
