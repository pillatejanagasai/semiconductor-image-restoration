# Architecture Details

## Model Architecture

The multi-task restoration model is based on an encoder-decoder architecture with task-specific branches.

### Shared Encoder
The shared encoder uses residual blocks combined with Convolutional Block Attention Module (CBAM) to extract robust, multi-scale features from the degraded semiconductor images.
- **Layers:** 3 downsampling stages using strided convolutions.
- **Attention:** CBAM applied at each stage to emphasize informative features across spatial and channel dimensions.
- **Output:** Multi-scale feature maps (skip connections) and a deep bottleneck representation.

### Defect Preservation Module
To ensure sub-nanometer defects are not smoothed out during restoration, this module computes a spatial attention mask.
- **Mechanism:** Applies an hourglass-like bottleneck with a sigmoid activation to output a mask values in `[0, 1]`.
- **Integration:** The mask is multiplied element-wise with the encoder features before passing to the decoders.

### Task-Specific Decoders
Three independent decoders upsample the features:
1. **Denoising Decoder:** Reconstructs the clean image at the original resolution.
2. **Deblurring Decoder:** Reconstructs the sharp image at the original resolution.
3. **Super-Resolution Decoder:** Reconstructs the image at 2x resolution (if configured) using sub-pixel convolutions (PixelShuffle).

## Loss Functions

The total loss is a weighted sum of multiple components:

$L_{total} = \lambda_{L1} L_{L1} + \lambda_{SSIM} L_{SSIM} + \lambda_{Perceptual} L_{Perceptual} + \lambda_{Edge} L_{Edge}$

- **L1 Loss:** For pixel-level reconstruction.
- **SSIM Loss:** For structural similarity.
- **Perceptual Loss:** Using a pre-trained VGG network to ensure perceptual quality.
- **Edge Loss:** Gradient-based loss to preserve sharp transitions and high-frequency defect patterns.

## Training Strategy
- **Mixed Precision:** Uses `torch.cuda.amp` to reduce memory usage and accelerate training.
- **Optimizer:** AdamW with a learning rate scheduler (CosineAnnealingLR).
- **Augmentations:** Random crops, rotations, flips, and intensity scaling.
