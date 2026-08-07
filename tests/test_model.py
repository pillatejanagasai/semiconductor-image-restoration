import torch
import pytest
from src.models.blocks import ResidualBlock, ChannelAttention, CBAM
from src.models.architecture import SharedEncoder, TaskSpecificDecoder, MultiTaskRestorationModel, DefectPreservationModule

def test_residual_block_shape():
    x = torch.randn(2, 64, 32, 32)
    block = ResidualBlock(64)
    out = block(x)
    assert out.shape == x.shape

def test_channel_attention_shape():
    x = torch.randn(2, 64, 32, 32)
    ca = ChannelAttention(64)
    out = ca(x)
    assert out.shape == x.shape

def test_cbam_shape():
    x = torch.randn(2, 64, 32, 32)
    cbam = CBAM(64)
    out = cbam(x)
    assert out.shape == x.shape

def test_encoder_output_shapes():
    x = torch.randn(2, 1, 64, 64)
    encoder = SharedEncoder(in_channels=1, base_channels=32)
    out, skips = encoder(x)
    assert out.shape == (2, 256, 8, 8)
    assert len(skips) == 3
    assert skips[0].shape == (2, 32, 64, 64)
    assert skips[1].shape == (2, 64, 32, 32)
    assert skips[2].shape == (2, 128, 16, 16)

def test_decoder_output_shape():
    encoder_out = torch.randn(2, 256, 8, 8)
    skips = [
        torch.randn(2, 32, 64, 64),
        torch.randn(2, 64, 32, 32),
        torch.randn(2, 128, 16, 16)
    ]
    decoder = TaskSpecificDecoder(base_channels=32, out_channels=1)
    out = decoder(encoder_out, skips)
    assert out.shape == (2, 1, 64, 64)

def test_multi_task_model_forward():
    x = torch.randn(2, 1, 64, 64)
    model = MultiTaskRestorationModel(in_channels=1, base_channels=16)
    out = model(x)
    assert isinstance(out, dict)
    assert 'denoised' in out
    assert 'deblurred' in out
    assert 'super_res' in out
    assert out['denoised'].shape == (2, 1, 64, 64)
    assert out['deblurred'].shape == (2, 1, 64, 64)
    assert out['super_res'].shape == (2, 1, 64, 64)

def test_model_parameter_count():
    model = MultiTaskRestorationModel(in_channels=1, base_channels=16)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 1000 < params < 10000000

def test_defect_preservation_forward():
    x = torch.randn(2, 64, 32, 32)
    module = DefectPreservationModule(in_channels=64)
    out, mask = module(x)
    assert out.shape == x.shape
    assert mask.shape == (2, 1, 32, 32)
    assert torch.all((mask >= 0) & (mask <= 1))
