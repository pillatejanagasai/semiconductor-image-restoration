import time
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

logger = logging.getLogger(__name__)

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR using torchmetrics."""
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(pred.device)
    return psnr_metric(pred, target).item()

def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM using torchmetrics."""
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    return ssim_metric(pred, target).item()

def compute_lpips(pred: torch.Tensor, target: torch.Tensor, device: str = 'cpu') -> float:
    """Compute LPIPS using the lpips library."""
    try:
        import lpips
    except ImportError:
        logger.warning("lpips library not found. Returning 0.0")
        return 0.0
        
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    if pred.size(1) == 1:
        pred = pred.repeat(1, 3, 1, 1)
    if target.size(1) == 1:
        target = target.repeat(1, 3, 1, 1)
        
    pred = pred * 2.0 - 1.0
    target = target * 2.0 - 1.0
    
    with torch.no_grad():
        d = loss_fn(pred, target)
        
    return d.mean().item()

def measure_inference_time(model: nn.Module, input_size: Tuple[int,...], device: str = 'cuda', num_runs: int = 100, warmup: int = 10) -> Dict[str, float]:
    """Measure inference time with warmup."""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            
        times = []
        for _ in range(num_runs):
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            _ = model(dummy_input)
            
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000) # ms
            
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times))
    }

def measure_memory_usage(model: nn.Module, input_size: Tuple[int,...], device: str = 'cuda') -> Dict[str, float]:
    """Measure peak GPU memory usage."""
    if device != 'cuda' or not torch.cuda.is_available():
        return {'peak_memory_mb': 0.0}
        
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(dummy_input)
        
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return {'peak_memory_mb': peak_memory}

def compute_gflops(model: nn.Module, input_size: Tuple[int,...]) -> float:
    """Compute GFLOPs using ptflops."""
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        logger.warning("ptflops library not found. Returning 0.0")
        return 0.0
        
    macs, params = get_model_complexity_info(
        model, tuple(input_size[1:]), as_strings=False, print_per_layer_stat=False
    )
    gflops = 2 * macs / (10**9)
    return gflops

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params
    }

class EvaluationSuite:
    """Comprehensive evaluation suite for restoration models."""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def evaluate_quality(self, pred, target):
        """Compute all quality metrics: PSNR, SSIM, LPIPS."""
        pred = pred.to(self.device)
        target = target.to(self.device)
        
        return {
            'psnr': compute_psnr(pred, target),
            'ssim': compute_ssim(pred, target),
            'lpips': compute_lpips(pred, target, self.device)
        }
    
    def evaluate_efficiency(self, model, input_size=(1,1,256,256)):
        """Compute efficiency metrics: time, memory, GFLOPs, params."""
        efficiency = {}
        efficiency.update(measure_inference_time(model, input_size, self.device))
        efficiency.update(measure_memory_usage(model, input_size, self.device))
        efficiency['gflops'] = compute_gflops(model, input_size)
        efficiency.update(count_parameters(model))
        return efficiency
    
    def full_evaluation(self, model, pred, target, input_size=(1,1,256,256)):
        """Run complete evaluation."""
        results = {}
        results['quality'] = self.evaluate_quality(pred, target)
        results['efficiency'] = self.evaluate_efficiency(model, input_size)
        return results
    
    def generate_report(self, results):
        """Generate formatted evaluation report string."""
        report = []
        report.append("="*40)
        report.append("EVALUATION REPORT")
        report.append("="*40)
        
        if 'quality' in results:
            report.append("\nQuality Metrics:")
            for k, v in results['quality'].items():
                report.append(f"  - {k.upper()}: {v:.4f}")
                
        if 'efficiency' in results:
            report.append("\nEfficiency Metrics:")
            eff = results['efficiency']
            report.append(f"  - Inference Time: {eff.get('mean_ms', 0):.2f} ms")
            report.append(f"  - Peak Memory: {eff.get('peak_memory_mb', 0):.2f} MB")
            report.append(f"  - GFLOPs: {eff.get('gflops', 0):.2f}")
            report.append(f"  - Parameters: {eff.get('total', 0)/1e6:.2f} M")
            
        report.append("="*40)
        return "\n".join(report)
