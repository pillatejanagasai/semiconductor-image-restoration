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
            
        losses = {}
        if self.weights.get('l1', 0) > 0: losses['l1'] = L1Loss()
        if self.weights.get('ssim', 0) > 0: losses['ssim'] = SSIMLoss()
        if self.weights.get('perceptual', 0) > 0: losses['perceptual'] = PerceptualLoss()
        if self.weights.get('edge', 0) > 0: losses['edge'] = EdgeLoss()
        if self.weights.get('frequency', 0) > 0: losses['frequency'] = FrequencyLoss()
        if self.weights.get('gradient', 0) > 0: losses['gradient'] = GradientLoss()
        
        self.losses = nn.ModuleDict(losses)
        
    def forward(self, pred, target):
        if isinstance(pred, dict):
            pred = pred.get('output', pred)
            
        total_loss = 0.0
        components = {}
        
        for name, loss_fn in self.losses.items():
            loss_val = loss_fn(pred, target)
            weighted_loss = self.weights[name] * loss_val
            total_loss += weighted_loss
            components[name] = weighted_loss.item()
                
        return total_loss, components
