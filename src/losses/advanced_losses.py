import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    """Edge-preserving loss using Sobel filter."""
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)
        self.l1_loss = nn.L1Loss()
        
    def _apply_sobel(self, x):
        b, c, h, w = x.shape
        x_reshaped = x.view(b * c, 1, h, w)
        
        grad_x = F.conv2d(x_reshaped, self.kernel_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.kernel_y, padding=1)
        
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return magnitude.view(b, c, h, w)

    def forward(self, pred, target):
        pred_edges = self._apply_sobel(pred)
        target_edges = self._apply_sobel(target)
        return self.l1_loss(pred_edges, target_edges)

class FrequencyLoss(nn.Module):
    """Frequency domain loss using FFT."""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")
        
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        return self.l1_loss(pred_mag, target_mag)

class GradientLoss(nn.Module):
    """Image gradient loss."""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred, target):
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        loss_x = self.l1_loss(pred_grad_x, target_grad_x)
        loss_y = self.l1_loss(pred_grad_y, target_grad_y)
        
        return loss_x + loss_y
