import torch
import torch.nn as nn
from .blocks import ResidualBlock, CBAM

class EncoderBlock(nn.Module):
    """Single encoder level: ResidualBlocks + Attention + Downsample."""
    def __init__(self, in_ch, out_ch, num_res_blocks=2, attention_type='cbam'):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(out_ch))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        if attention_type == 'cbam':
            self.attention = CBAM(out_ch)
        else:
            self.attention = nn.Identity()
            
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.res_blocks(x)
        features_before_pool = self.attention(x)
        features_after_pool = self.pool(features_before_pool)
        return features_before_pool, features_after_pool

class SharedEncoder(nn.Module):
    """4-level U-Net encoder shared across all restoration tasks."""
    def __init__(self, in_channels=1, encoder_channels=[64, 128, 256, 512], num_res_blocks=2, attention_type='cbam'):
        super().__init__()
        self.levels = nn.ModuleList()
        
        # 4 EncoderBlocks with increasing channels
        current_in_channels = in_channels
        for out_channels in encoder_channels:
            self.levels.append(EncoderBlock(current_in_channels, out_channels, num_res_blocks, attention_type))
            current_in_channels = out_channels
            
        # Bottleneck: ResidualBlock at deepest level
        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_in_channels, current_in_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(current_in_channels * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(current_in_channels * 2),
            ResidualBlock(current_in_channels * 2)
        )
        
    def forward(self, x):
        skip_features = []
        for level in self.levels:
            skip_feat, x = level(x)
            skip_features.append(skip_feat)
            
        bottleneck_feat = self.bottleneck(x)
        return skip_features, bottleneck_feat
