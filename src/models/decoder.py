import torch
import torch.nn as nn
from .blocks import ResidualBlock, CBAM

class DecoderBlock(nn.Module):
    """Single decoder level: Upsample + Concat skip + ResidualBlocks + Attention."""
    def __init__(self, in_ch, skip_ch, out_ch, num_res_blocks=2, attention_type='cbam'):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        
        # After concat with skip connection
        conv_in_ch = (in_ch // 2) + skip_ch
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(conv_in_ch, out_ch, kernel_size=3, padding=1, bias=False),
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
            
    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle padding if dimensions don't match exactly
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        if diffY > 0 or diffX > 0:
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            
        x = torch.cat([skip, x], dim=1)
        x = self.conv_reduce(x)
        x = self.res_blocks(x)
        x = self.attention(x)
        return x

class TaskDecoder(nn.Module):
    """Decoder for a specific restoration task."""
    def __init__(self, encoder_channels=[64, 128, 256, 512], out_channels=1, num_res_blocks=2, attention_type='cbam'):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        bottleneck_channels = encoder_channels[-1] * 2
        
        # 4 DecoderBlocks (symmetric to encoder)
        reversed_channels = list(reversed(encoder_channels))
        
        current_in_channels = bottleneck_channels
        for i, skip_ch in enumerate(reversed_channels):
            out_ch = reversed_channels[i]
            self.blocks.append(DecoderBlock(current_in_channels, skip_ch, out_ch, num_res_blocks, attention_type))
            current_in_channels = out_ch
            
        # Final 1x1 conv to out_channels
        self.final_conv = nn.Conv2d(current_in_channels, out_channels, kernel_size=1)
        
    def forward(self, bottleneck, skip_features):
        x = bottleneck
        # skip_features are in order from shallow to deep, we need them reversed
        for i, block in enumerate(self.blocks):
            skip = skip_features[-(i + 1)]
            x = block(x, skip)
            
        out = self.final_conv(x)
        return out
