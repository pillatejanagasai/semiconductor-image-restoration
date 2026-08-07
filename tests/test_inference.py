import os
import torch
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from src.models.architecture import MultiTaskRestorationModel

def test_onnx_export(tmp_path):
    model = MultiTaskRestorationModel(in_channels=1, base_channels=16)
    model.eval()
    dummy_input = torch.randn(1, 1, 64, 64)
    export_path = tmp_path / "model.onnx"
    torch.onnx.export(model, dummy_input, str(export_path), opset_version=11)
    assert export_path.exists()

def test_inference_engine_init(tmp_path):
    model = MultiTaskRestorationModel(in_channels=1, base_channels=16)
    ckpt_path = tmp_path / "best_model.pth"
    torch.save(model.state_dict(), ckpt_path)
    
    model_loaded = MultiTaskRestorationModel(in_channels=1, base_channels=16)
    model_loaded.load_state_dict(torch.load(ckpt_path))
    model_loaded.eval()
    
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        out = model_loaded(x)
    assert 'denoised' in out

def test_preprocess_postprocess():
    img = np.random.rand(64, 64).astype(np.float32)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    out_np = tensor.squeeze().numpy()
    assert np.allclose(img, out_np)
