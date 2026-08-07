import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class L1Loss(nn.Module):
    """Standard L1 (Mean Absolute Error) loss."""
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
        
    def forward(self, pred, target):
        return self.loss(pred, target)

class SSIMLoss(nn.Module):
    """Differentiable SSIM loss."""
    def __init__(self, window_size=11, channels=1):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.window = self._create_gaussian_window(window_size, channels)
        
    def _create_gaussian_window(self, window_size, channels):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2):
        if img1.size(-3) != self.channels:
            self.channels = img1.size(-3)
            self.window = self._create_gaussian_window(self.window_size, self.channels).to(img1.device)
        else:
            self.window = self.window.to(img1.device)
            
        window = self.window
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channels)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=self.channels) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def forward(self, pred, target):
        return 1 - self._ssim(pred, target)

class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss."""
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3'], weights=None):
        super().__init__()
        import torchvision.models as models
        
        vgg = models.vgg16(pretrained=True).features
        self.layer_names = ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3']
        self.target_layers = layers
        self.weights = weights if weights else [1.0] * len(layers)
        
        self.slice_blocks = nn.ModuleList()
        current_block = nn.Sequential()
        block_idx = 0
        layer_idx = 0
        
        for name, module in vgg.named_children():
            current_block.add_module(name, module)
            if isinstance(module, nn.ReLU):
                if self.layer_names[layer_idx] in self.target_layers:
                    self.slice_blocks.append(current_block)
                    current_block = nn.Sequential()
                    block_idx += 1
                layer_idx += 1
            if block_idx == len(self.target_layers):
                break
                
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
        loss = 0.0
        
        # Handle single-channel input by repeating to 3 channels
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)
            
        x = pred
        y = target
        
        for i, block in enumerate(self.slice_blocks):
            x = block(x)
            y = block(y)
            loss += self.weights[i] * F.l1_loss(x, y)
            
        return loss
