import torch
from src.models.restoration_model import MultiTaskRestorationNet
from src.losses.combined_loss import CombinedLoss

def test_model_forward():
    model = MultiTaskRestorationNet(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert isinstance(out, dict)
    assert 'output' in out
    assert out['output'].shape == (2, 1, 64, 64)

def test_combined_loss():
    loss_fn = CombinedLoss(weights={'l1': 1.0, 'ssim': 0.5, 'lpips': 0.1})
    pred = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64)
    
    # Test tensor input
    loss_val, components = loss_fn(pred, target)
    assert isinstance(loss_val, torch.Tensor)
    assert isinstance(components, dict)
    assert 'l1' in components
    assert 'ssim' in components
    assert 'lpips' in components
    
    # Test dict input (like the model outputs)
    pred_dict = {'output': pred}
    loss_val_dict, components_dict = loss_fn(pred_dict, target)
    assert torch.allclose(loss_val, loss_val_dict)
