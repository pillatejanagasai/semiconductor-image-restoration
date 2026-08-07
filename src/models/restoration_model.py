import torch
import torch.nn as nn
from .encoder import SharedEncoder
from .decoder import TaskDecoder
from .defect_preservation import DefectPreservationModule
from .blocks import FeatureFusion

class MultiTaskRestorationNet(nn.Module):
    """Multi-task semiconductor image restoration network.
    
    Architecture:
    - Shared encoder (4-level U-Net)
    - 3 task-specific decoders (denoise, deblur, super_resolve)
    - Feature fusion across tasks
    - Optional defect preservation module
    """
    def __init__(self, in_channels=1, out_channels=1, encoder_channels=[64, 128, 256, 512], 
                 num_residual_blocks=2, attention_type='cbam', num_tasks=3, 
                 task_names=['denoise', 'deblur', 'super_resolve'], use_defect_preservation=True):
        super().__init__()
        self.task_names = task_names
        self.use_defect_preservation = use_defect_preservation
        
        # Shared Encoder
        self.shared_encoder = SharedEncoder(in_channels, encoder_channels, num_residual_blocks, attention_type)
        
        # Task Decoders
        self.task_decoders = nn.ModuleDict()
        for name in task_names:
            self.task_decoders[name] = TaskDecoder(encoder_channels, out_channels, num_residual_blocks, attention_type)
            
        # Feature Fusion
        # Fuse outputs from all task decoders. Each outputs `out_channels`
        self.feature_fusion = FeatureFusion([out_channels] * len(task_names), out_channels)
        
        # Defect Preservation
        if use_defect_preservation:
            # We extract features from the first level skip connection for defect attention
            self.defect_preservation = DefectPreservationModule(in_channels=in_channels, feature_channels=encoder_channels[0])
            
        # Final Reconstruction
        self.final_reconstruction = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # Assuming output range [0, 1]
        )
        
    def forward(self, x):
        results = {}
        
        # 1. Encode
        skip_features, bottleneck = self.shared_encoder(x)
        
        # 2. Decode each task
        task_outputs = []
        for name in self.task_names:
            out = self.task_decoders[name](bottleneck, skip_features)
            results[name] = out
            task_outputs.append(out)
            
        # 3. Fuse task outputs
        fused_features = self.feature_fusion(task_outputs)
        reconstructed = self.final_reconstruction(fused_features)
        
        # 4. Apply defect preservation if enabled
        if self.use_defect_preservation:
            # use the shallowest skip features
            shallow_features = skip_features[0] 
            final_output, defect_mask = self.defect_preservation(x, reconstructed, shallow_features)
            results['defect_mask'] = defect_mask
        else:
            final_output = reconstructed
            
        # 5. Return dict
        results['output'] = final_output
        return results
