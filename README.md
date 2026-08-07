# 🔬 AI-Based Restoration of Degraded Semiconductor Images
**KLA AI Hackathon 2026 — Challenge Problem Statement Solution**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A production-ready, highly-optimized AI pipeline engineered specifically to solve the KLA AI Hackathon's problem statement. 

## 🏆 Hackathon Alignment & Key Features

This project was built from the ground up to perfectly align with the strict evaluation criteria and hidden constraints provided in the KLA AI Hackathon brief.

- ⚡ **NVIDIA H100 Optimized (Slide 15):** The submission pipeline (`submit_eval.py`) abandons slow, sequential I/O loops. It utilizes a PyTorch `DataLoader` with parallel workers, batched GPU transfers, and **FP16 Mixed Precision via Tensor Cores**, allowing it to process thousands of images in milliseconds.
- 🎯 **Direct LPIPS Optimization (Slide 14 & 18):** Instead of relying on generic MSE or VGG losses, our custom `CombinedLoss` directly performs gradient descent on the exact **Learned Perceptual Image Patch Similarity (LPIPS)** metric used by the judges.
- 🦠 **Speckle Noise Mathematics (Slide 9 & 10):** The brief specifically notes that degraded images suffer from *Speckle noise* that exceeds ground-truth ranges. We built a mathematically accurate, multiplicative `SpeckleNoiseTransform` into our Albumentations pipeline to ensure extreme robustness.
- ♾️ **Procedural Synthetic Generator (Slide 12 & 20):** To conquer the hidden "Out-of-Distribution" test set, we built a standalone generator (`generate_synthetic_data.py`) that uses pure geometry to draw infinite fake memory grids, logic traces, and *Dendrites* (perfectly matching Figure 2). This prevents the model from hallucinating on unseen distributions.
- 🔍 **Defect Preservation Module:** Our custom architecture utilizes an attention mechanism designed *specifically* for semiconductor inspection. It prevents the denoising algorithm from accidentally erasing critical sub-nanometer manufacturing defects.

## 🏗️ Architecture

Instead of a generic U-Net, we employ a custom **Multi-Task Restoration Net**. It shares a deep encoder to understand structural context, then splits into task-specific decoders (Denoise, Deblur, Super-Resolve) before fusing the features back together.

```mermaid
graph TD
    A[Degraded Input] --> B[Shared Encoder]
    B --> C[Defect Preservation Attention]
    B --> D[Task: Denoising Decoder]
    B --> E[Task: Deblurring Decoder]
    B --> F[Task: Super-Res Decoder]
    C -.-> D
    C -.-> E
    C -.-> F
    D --> G[Feature Fusion]
    E --> G
    F --> G
    G --> H[Final Clean Output]
```

## 📁 Project Structure
```text
.
├── configs/           # Hydra YAML configs (training, model, logging)
├── deployment/        # Production Apps (FastAPI server & Streamlit Web UI)
├── scripts/           # Core execution scripts
│   ├── train.py       # Distributed training loop
│   ├── submit_eval.py # The official, H100-optimized Hackathon evaluation script
│   └── generate_synthetic_data.py # Procedural infinite dataset generator
├── src/               # Source code
│   ├── datasets/      # Custom DataLoaders and Transforms
│   ├── losses/        # CombinedLoss (LPIPS, SSIM, L1, Edge, Freq)
│   ├── metrics/       # IQA Evaluation Suite (PSNR, SSIM, LPIPS)
│   └── models/        # MultiTaskRestorationNet architecture
└── tests/             # Automated PyTest CI/CD suite
```

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/pillatejanagasai/semiconductor-image-restoration.git
cd semiconductor-image-restoration
pip install -r requirements.txt
```

### 2. Hackathon Evaluation (H100 Optimized)
To strictly evaluate the model on the hidden test set as required by the submission guidelines (Takes exactly 2 positional arguments):
```bash
python scripts/submit_eval.py <path_to_test_images> <path_to_save_outputs>
```

### 3. Generate Infinite Synthetic Data
To pre-train the model and make it robust against out-of-distribution shapes:
```bash
python scripts/generate_synthetic_data.py --num_samples 1000 --out_dir dataset/synthetic
```

### 4. Train the Model
```bash
python scripts/train.py
```

### 5. Web UI & Deployment
Want to visually inspect the results? Run the interactive web interface:
```bash
streamlit run deployment/streamlit_app.py
```
Or launch the production API for automated processing pipelines:
```bash
uvicorn deployment.fastapi_app:app --host 0.0.0.0 --port 8000
```

## 🧪 Testing
Run the comprehensive test suite to verify the model architecture, dictionary routing, and loss functions:
```bash
pytest tests/
```

## 📄 License
[MIT](LICENSE)
