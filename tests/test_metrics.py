import torch
import pytest
from src.metrics.evaluation import compute_psnr, compute_ssim, compute_niqe
from src.models.architecture import MultiTaskRestorationModel

def test_compute_psnr():
    x = torch.ones(1, 1, 32, 32)
    y = torch.ones(1, 1, 32, 32)
    psnr = compute_psnr(x, y, data_range=1.0)
    assert psnr > 80.0 or torch.isinf(torch.tensor(psnr))

def test_compute_ssim():
    x = torch.ones(1, 1, 32, 32)
    y = torch.ones(1, 1, 32, 32)
    ssim = compute_ssim(x, y, data_range=1.0)
    assert pytest.approx(ssim, 0.01) == 1.0

def test_compute_psnr_different():
    x = torch.ones(1, 1, 32, 32)
    y = torch.zeros(1, 1, 32, 32)
    psnr = compute_psnr(x, y, data_range=1.0)
    assert psnr < float('inf')

def test_count_parameters():
    model = MultiTaskRestorationModel(in_channels=1, base_channels=16)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params > 0

def test_evaluation_suite():
    x = torch.rand(1, 1, 64, 64)
    y = torch.rand(1, 1, 64, 64)
    psnr = compute_psnr(x, y, data_range=1.0)
    ssim = compute_ssim(x, y, data_range=1.0)
    assert isinstance(psnr, float)
    assert isinstance(ssim, float)
