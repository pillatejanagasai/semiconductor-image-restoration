import os
import sys

# Ensure the root project directory is in the PYTHONPATH so we can import 'src'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import logging
import random
import numpy as np
from src.datasets import SEMDataset
from src.datasets.transforms import get_training_transforms, get_validation_transforms, get_degradation_transforms
from src.models import MultiTaskRestorationNet
from src.losses import CombinedLoss
from src.trainers import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    set_seed(cfg.project.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    train_transforms = get_training_transforms(cfg.dataset.patch_size)
    val_transforms = get_validation_transforms(cfg.dataset.patch_size)
    deg_transforms = get_degradation_transforms()
    
    # Use synthetic data if available, otherwise raw
    data_path = "dataset/synthetic" if os.path.exists("dataset/synthetic/clean") else cfg.dataset.data_dir
    
    train_dataset = SEMDataset(
        root_dir=data_path,
        split='',  # Since synthetic generator doesn't create train/val splits yet
        paired_mode=True, # Synthetic generator creates paired data
        transform=train_transforms,
        degradation_transform=deg_transforms,
        patch_size=cfg.dataset.patch_size
    )
    
    val_dataset = SEMDataset(
        root_dir=data_path,
        split='',
        paired_mode=True,
        transform=val_transforms,
        degradation_transform=deg_transforms,
        patch_size=cfg.dataset.patch_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True
    )
    
    model = MultiTaskRestorationNet(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        encoder_channels=cfg.model.encoder_channels,
        num_residual_blocks=cfg.model.num_residual_blocks,
        attention_type=cfg.model.attention_type,
        num_tasks=cfg.model.num_tasks,
        task_names=cfg.model.task_names,
        use_defect_preservation=cfg.model.use_defect_preservation
    )
    
    criterion = CombinedLoss(
        weights=cfg.training.loss_weights
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    
    trainer_config = {
        'use_amp': cfg.training.mixed_precision,
        'scheduler': cfg.training.scheduler,
        'log_dir': cfg.training.logging.log_dir,
        'checkpoint_dir': cfg.training.checkpoint.save_dir,
        'save_every_n': cfg.training.checkpoint.save_every_n_epochs,
        'patience': cfg.training.early_stopping.patience,
        'epochs': cfg.training.epochs,
        'max_grad_norm': cfg.training.gradient_clip_max_norm,
        'log_every_n_steps': cfg.training.logging.log_every_n_steps
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=trainer_config,
        device=device
    )
    
    if cfg.training.get('resume_from'):
        trainer.load_checkpoint(cfg.training.resume_from)
        
    logger.info("Starting training...")
    trainer.fit()
    logger.info("Training completed.")

if __name__ == '__main__':
    main()
