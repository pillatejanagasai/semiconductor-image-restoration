import os
import torch
import logging

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""
    def __init__(self, patience=20, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, metric) -> bool:
        if self.best_metric is None:
            self.best_metric = metric
        elif self.mode == 'min' and metric > self.best_metric - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'max' and metric < self.best_metric + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_metric = metric
            self.counter = 0
        return self.early_stop

    def reset(self):
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

class ModelCheckpoint:
    """Save model checkpoints during training."""
    def __init__(self, save_dir, save_best=True, save_latest=True, save_every_n=10, mode='min'):
        self.save_dir = save_dir
        self.save_best = save_best
        self.save_latest = save_latest
        self.save_every_n = save_every_n
        self.mode = mode
        self.best_metric = None
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, epoch, metric, model, optimizer, scheduler=None, scaler=None, extra=None):
        is_best = False
        if self.best_metric is None:
            self.best_metric = metric
            is_best = True
        elif self.mode == 'min' and metric < self.best_metric:
            self.best_metric = metric
            is_best = True
        elif self.mode == 'max' and metric > self.best_metric:
            self.best_metric = metric
            is_best = True

        if is_best and self.save_best:
            # Only save model weights for best model to save disk space
            self._save(os.path.join(self.save_dir, 'best_model.pt'), epoch, model, extra=extra)

        if self.save_latest:
            # Save full training state for latest model so training can be resumed
            self._save(os.path.join(self.save_dir, 'latest_model.pt'), epoch, model, optimizer, scheduler, scaler, extra)

        if self.save_every_n > 0 and epoch % self.save_every_n == 0:
            # Delete previous epoch checkpoint to save disk space
            prev_epoch = epoch - self.save_every_n
            prev_path = os.path.join(self.save_dir, f'model_epoch_{prev_epoch}.pt')
            if os.path.exists(prev_path):
                try:
                    os.remove(prev_path)
                except OSError:
                    pass
            # Only save model weights for periodic checkpoints
            self._save(os.path.join(self.save_dir, f'model_epoch_{epoch}.pt'), epoch, model, extra=extra)

    def _save(self, filepath, epoch, model, optimizer=None, scheduler=None, scaler=None, extra=None):
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }
        if optimizer is not None:
            state['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
        if scaler is not None:
            state['scaler_state_dict'] = scaler.state_dict()
        if extra is not None:
            state['extra'] = extra
        
        torch.save(state, filepath)
        logger.debug(f"Saved checkpoint to {filepath}")

class LearningRateLogger:
    """Log learning rate to TensorBoard."""
    def __init__(self, writer):
        self.writer = writer

    def __call__(self, optimizer, epoch, step):
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f'LR/group_{i}', param_group['lr'], step)
