import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import logging
import random
import numpy as np
import os

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
    
    set_seed(cfg.get('seed', 42))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    train_transforms = get_training_transforms(cfg.dataset.img_size)
    val_transforms = get_validation_transforms(cfg.dataset.img_size)
    deg_transforms = get_degradation_transforms(cfg.dataset.degradation)
    
    train_dataset = SEMDataset(
        data_dir=cfg.dataset.train_dir,
        transform=train_transforms,
        degradation_transform=deg_transforms,
        is_train=True
    )
    
    val_dataset = SEMDataset(
        data_dir=cfg.dataset.val_dir,
        transform=val_transforms,
        degradation_transform=deg_transforms,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    
    model = MultiTaskRestorationNet(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        base_filters=cfg.model.base_filters
    )
    
    criterion = CombinedLoss(
        weights=cfg.loss.weights
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay
    )
    
    trainer_config = {
        'use_amp': cfg.training.use_amp,
        'scheduler': cfg.training.scheduler,
        'log_dir': cfg.training.log_dir,
        'checkpoint_dir': cfg.training.checkpoint_dir,
        'save_every_n': cfg.training.save_every_n,
        'patience': cfg.training.patience,
        'epochs': cfg.training.epochs,
        'max_grad_norm': cfg.training.max_grad_norm,
        'log_every_n_steps': cfg.training.log_every_n_steps
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
