# API Reference

## src.datasets
- `SEMDataset`: Custom PyTorch Dataset for loading synthetic and paired semiconductor images.
- `DataSplitter`: Utility for splitting datasets into train, validation, and test sets.

## src.models
- `SharedEncoder`: Feature extractor with CBAM attention.
- `TaskSpecificDecoder`: Decoder module for specific restoration tasks.
- `DefectPreservationModule`: Attention module to preserve critical defects.
- `MultiTaskRestorationModel`: The complete multi-task model.

## src.losses
- `CombinedLoss`: Weighted combination of L1, SSIM, Edge, and Perceptual losses.
- `L1Loss`, `SSIMLoss`, `EdgeLoss`, `PerceptualLoss`, `FrequencyLoss`, `GradientLoss`: Individual loss components.

## src.trainers
- `Trainer`: Handles the training loop, validation, logging, and checkpointing. Uses mixed precision.

## src.metrics
- `compute_psnr`: Computes Peak Signal-to-Noise Ratio.
- `compute_ssim`: Computes Structural Similarity Index.
- `compute_niqe`: Computes Natural Image Quality Evaluator.

## src.utils.preprocessing
- Functions for histogram equalization, CLAHE, normalization, and hot pixel removal.
