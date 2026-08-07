import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelEdgeDetector(nn.Module):
    """Differentiable Sobel edge detection."""
    def __init__(self):
        super().__init__()
        # Register Sobel kernels as non-learnable buffers
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)
        
    def forward(self, x):
        # Handle multiple channels by applying independently
        b, c, h, w = x.shape
        x_reshaped = x.view(b * c, 1, h, w)
        
        grad_x = F.conv2d(x_reshaped, self.kernel_x, padding=1)
        grad_y = F.conv2d(x_reshaped, self.kernel_y, padding=1)
        
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return magnitude.view(b, c, h, w)

class GradientMapExtractor(nn.Module):
    """Extract multi-directional gradient maps."""
    def __init__(self):
        super().__init__()
        # 4 directional gradient kernels (horizontal, vertical, 2 diagonals)
        k_h = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        k_v = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        k_d1 = torch.tensor([[-1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        k_d2 = torch.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('k_h', k_h)
        self.register_buffer('k_v', k_v)
        self.register_buffer('k_d1', k_d1)
        self.register_buffer('k_d2', k_d2)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_reshaped = x.view(b * c, 1, h, w)
        
        g_h = F.conv2d(x_reshaped, self.k_h, padding=1)
        g_v = F.conv2d(x_reshaped, self.k_v, padding=1)
        g_d1 = F.conv2d(x_reshaped, self.k_d1, padding=1)
        g_d2 = F.conv2d(x_reshaped, self.k_d2, padding=1)
        
        g_h = g_h.view(b, c, h, w)
        g_v = g_v.view(b, c, h, w)
        g_d1 = g_d1.view(b, c, h, w)
        g_d2 = g_d2.view(b, c, h, w)
        
        return torch.cat([g_h, g_v, g_d1, g_d2], dim=1)

class DefectAttentionModule(nn.Module):
    """Learns to attend to defect-like features in SEM images."""
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        # Takes edge + gradient features as input
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features, edge_map, gradient_map):
        x = torch.cat([features, edge_map, gradient_map], dim=1)
        attention_mask = self.conv(x)
        return attention_mask

class DefectPreservationModule(nn.Module):
    """Complete defect preservation module."""
    def __init__(self, in_channels=1, feature_channels=64):
        super().__init__()
        self.edge_detector = SobelEdgeDetector()
        self.gradient_extractor = GradientMapExtractor()
        
        # edge (in_channels) + gradients (4 * in_channels) + features (feature_channels)
        attention_in_channels = in_channels + (4 * in_channels) + feature_channels
        self.attention_module = DefectAttentionModule(attention_in_channels)
        
    def forward(self, original_input, restored_output, features):
        edge_map = self.edge_detector(original_input)
        gradient_map = self.gradient_extractor(original_input)
        
        defect_mask = self.attention_module(features, edge_map, gradient_map)
        
        blended_output = restored_output * (1 - defect_mask) + original_input * defect_mask
        
        return blended_output, defect_mask
