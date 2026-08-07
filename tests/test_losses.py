import torch
import pytest
from src.losses.loss_functions import L1Loss, SSIMLoss, PerceptualLoss, EdgeLoss, FrequencyLoss, GradientLoss, CombinedLoss

def test_l1_loss():
    x = torch.ones(2, 1, 32, 32)
    y = torch.zeros(2, 1, 32, 32)
    loss_fn = L1Loss()
    loss = loss_fn(x, y)
    assert torch.isclose(loss, torch.tensor(1.0))

def test_ssim_loss_identical():
    x = torch.randn(2, 1, 32, 32)
    loss_fn = SSIMLoss(window_size=11)
    loss = loss_fn(x, x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)

def test_ssim_loss_different():
    x = torch.ones(2, 1, 32, 32)
    y = torch.zeros(2, 1, 32, 32)
    loss_fn = SSIMLoss(window_size=11)
    loss = loss_fn(x, y)
    assert loss.item() > 0.0

def test_perceptual_loss_forward():
    x = torch.randn(2, 3, 64, 64)
    y = torch.randn(2, 3, 64, 64)
    loss_fn = PerceptualLoss()
    loss = loss_fn(x, y)
    assert loss.item() > 0.0
    assert loss.shape == torch.Size([])

def test_edge_loss_forward():
    x = torch.randn(2, 1, 32, 32, requires_grad=True)
    y = torch.randn(2, 1, 32, 32)
    loss_fn = EdgeLoss()
    loss = loss_fn(x, y)
    assert loss.item() >= 0.0
    loss.backward()
    assert x.grad is not None

def test_frequency_loss_forward():
    x = torch.randn(2, 1, 32, 32)
    y = torch.randn(2, 1, 32, 32)
    loss_fn = FrequencyLoss()
    loss = loss_fn(x, y)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0.0

def test_gradient_loss_forward():
    x = torch.randn(2, 1, 32, 32)
    y = torch.randn(2, 1, 32, 32)
    loss_fn = GradientLoss()
    loss = loss_fn(x, y)
    assert loss.shape == torch.Size([])
    assert loss.item() >= 0.0

def test_combined_loss():
    x = torch.randn(2, 1, 32, 32)
    y = torch.randn(2, 1, 32, 32)
    weights = {'l1': 1.0, 'ssim': 0.1}
    loss_fn = CombinedLoss(weights=weights)
    total_loss, components = loss_fn(x, y)
    assert total_loss.item() > 0.0
    assert 'l1' in components
    assert 'ssim' in components

def test_combined_loss_gradient_flow():
    x = torch.randn(2, 1, 32, 32, requires_grad=True)
    y = torch.randn(2, 1, 32, 32)
    loss_fn = CombinedLoss(weights={'l1': 1.0, 'edge': 0.5})
    total_loss, _ = loss_fn(x, y)
    total_loss.backward()
    assert x.grad is not None
