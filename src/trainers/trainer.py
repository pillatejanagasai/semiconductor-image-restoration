import os
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from .callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)

class Trainer:
    """Production-ready training engine for semiconductor image restoration.
    
    Features:
    - Mixed precision training (FP16)
    - Gradient clipping
    - Checkpoint saving/loading (best + latest + periodic)
    - Early stopping with configurable patience
    - Resume training from checkpoint
    - Learning rate scheduling (CosineAnnealingWarmRestarts)
    - TensorBoard logging (losses, metrics, images, LR)
    - Validation after each epoch
    """
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        self.scaler = torch.amp.GradScaler('cuda') if config.get('use_amp', True) and torch.cuda.is_available() else None
        
        if config.get('scheduler'):
            scheduler_cfg = config.get('scheduler_params', {})
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=scheduler_cfg.get('T_0', 50),
                T_mult=scheduler_cfg.get('T_mult', 1),
                eta_min=scheduler_cfg.get('eta_min', 1e-6)
            )
        else:
            self.scheduler = None

        log_dir = config.get('log_dir', 'runs/experiment')
        self.writer = SummaryWriter(log_dir=log_dir)
        
        save_dir = config.get('checkpoint_dir', 'checkpoints')
        self.checkpoint = ModelCheckpoint(
            save_dir=save_dir,
            save_best=True,
            save_latest=True,
            save_every_n=config.get('save_every_n', 10),
            mode='min'
        )
        
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 20),
            min_delta=config.get('min_delta', 1e-4),
            mode='min'
        )
        
        self.start_epoch = 0
        self.global_step = 0
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
    
    def train_epoch(self, epoch):
        """Train for one epoch. Returns dict of average losses."""
        self.model.train()
        total_loss = 0.0
        losses_dict = {}
        
        for batch_idx, batch in enumerate(self.train_loader):
            degraded = batch['degraded'].to(self.device)
            clean = batch['clean'].to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(degraded)
                    loss, loss_dict = self.criterion(outputs, clean)
                
                self.scaler.scale(loss).backward()
                
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(degraded)
                loss, loss_dict = self.criterion(outputs, clean)
                
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            total_loss += loss.item()
            for k, v in loss_dict.items():
                losses_dict[k] = losses_dict.get(k, 0.0) + (v.item() if hasattr(v, 'item') else v)
                
            if self.global_step % self.config.get('log_every_n_steps', 10) == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'Train/{k}', (v.item() if hasattr(v, 'item') else v), self.global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)
                
            self.global_step += 1
            
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_losses_dict = {k: v / num_batches for k, v in losses_dict.items()}
        avg_losses_dict['loss'] = avg_loss
        return avg_losses_dict
    
    def validate_epoch(self, epoch):
        """Validate for one epoch. Returns dict of average losses and metrics."""
        self.model.eval()
        total_loss = 0.0
        losses_dict = {}
        total_psnr = 0.0
        total_ssim = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                degraded = batch['degraded'].to(self.device)
                clean = batch['clean'].to(self.device)
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(degraded)
                        loss, loss_dict = self.criterion(outputs, clean)
                else:
                    outputs = self.model(degraded)
                    loss, loss_dict = self.criterion(outputs, clean)
                
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    losses_dict[k] = losses_dict.get(k, 0.0) + (v.item() if hasattr(v, 'item') else v)
                    
                pred_img = outputs['output'] if isinstance(outputs, dict) else outputs
                
                psnr = self._compute_psnr(pred_img, clean)
                ssim = self._compute_ssim_metric(pred_img, clean)
                
                total_psnr += psnr
                total_ssim += ssim
                
                if batch_idx == 0:
                    self._log_images(epoch, degraded, clean, pred_img)
                    
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_losses_dict = {k: v / num_batches for k, v in losses_dict.items()}
        avg_losses_dict['loss'] = avg_loss
        
        metrics = {
            'val_loss': avg_loss,
            'val_psnr': total_psnr / num_batches,
            'val_ssim': total_ssim / num_batches
        }
        
        for k, v in avg_losses_dict.items():
            self.writer.add_scalar(f'Val/{k}', v, epoch)
        for k, v in metrics.items():
            self.writer.add_scalar(f'Val/{k.split("_")[1]}', v, epoch)
            
        return metrics
    
    def fit(self):
        """Full training loop."""
        num_epochs = self.config.get('epochs', 100)
        
        for epoch in range(self.start_epoch, num_epochs):
            logger.info(f"Starting epoch {epoch}")
            train_losses = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(
                epoch=epoch,
                is_best=False # Handled internally by ModelCheckpoint
            )
            
            # Using our callback object for checkpointing
            self.checkpoint(
                epoch=epoch,
                metric=val_loss,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                extra={'config': self.config}
            )
            
            if self._check_early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
        self.writer.close()
    
    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """Save model checkpoint. (Direct save alternative to callback)"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
            
        save_path = os.path.join(self.config.get('checkpoint_dir', 'checkpoints'), filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler is not None:
            state['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(state, save_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training."""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found at {checkpoint_path}")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.start_epoch-1})")
    
    def _check_early_stopping(self, metric):
        """Check if training should stop early."""
        return self.early_stopping(metric)
    
    def _log_images(self, epoch, degraded, clean, restored):
        """Log image comparisons to TensorBoard."""
        import torchvision
        
        # Take up to 4 images
        n = min(4, degraded.size(0))
        degraded = degraded[:n]
        clean = clean[:n]
        restored = restored[:n]
        
        # Stack vertically: degraded, restored, clean
        grid = torch.cat([degraded, restored, clean], dim=0)
        img_grid = torchvision.utils.make_grid(grid, nrow=n, normalize=True, scale_each=True)
        self.writer.add_image('Images (Degraded, Restored, Clean)', img_grid, epoch)
    
    def _compute_psnr(self, pred, target):
        """Compute PSNR between prediction and target tensors."""
        mse = torch.nn.functional.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        psnr = 10 * math.log10(1.0 / mse.item())
        return psnr
    
    def _compute_ssim_metric(self, pred, target):
        """Compute SSIM between prediction and target tensors."""
        try:
            from torchmetrics.functional.image import structural_similarity_index_measure
            return structural_similarity_index_measure(pred, target).item()
        except ImportError:
            return 0.0 # Placeholder if torchmetrics not available
