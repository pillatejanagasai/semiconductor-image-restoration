import os
import glob
import logging
import shutil
import random
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class DataSplitter:
    """Utility class to split SEM image datasets into train, val, and test sets."""
    
    @staticmethod
    def split_dataset(source_dir, output_dir, split_ratios, seed=42, formats=None):
        random.seed(seed)
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        
        if not formats:
            formats = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
            
        is_paired = (source_path / 'clean').exists() and (source_path / 'degraded').exists()
        
        all_files = []
        if is_paired:
            clean_dir = source_path / 'clean'
            for ext in formats:
                all_files.extend(clean_dir.rglob(f"*{ext.lower()}"))
                all_files.extend(clean_dir.rglob(f"*{ext.upper()}"))
        else:
            for ext in formats:
                all_files.extend(source_path.rglob(f"*{ext.lower()}"))
                all_files.extend(source_path.rglob(f"*{ext.upper()}"))
                
        all_files = sorted(list(set(all_files)))
        
        if not all_files:
            logger.warning(f"No files found in {source_dir} matching formats {formats}")
            return {'train': 0, 'val': 0, 'test': 0}
            
        random.shuffle(all_files)
        
        total_files = len(all_files)
        train_end = int(total_files * split_ratios.get('train', 0.7))
        val_end = train_end + int(total_files * split_ratios.get('val', 0.15))
        
        splits = {
            'train': all_files[:train_end],
            'val': all_files[train_end:val_end],
            'test': all_files[val_end:]
        }
        
        result_counts = {}
        
        for split_name, files in splits.items():
            split_out_dir = output_path / split_name
            
            if is_paired:
                out_clean = split_out_dir / 'clean'
                out_degraded = split_out_dir / 'degraded'
                out_clean.mkdir(parents=True, exist_ok=True)
                out_degraded.mkdir(parents=True, exist_ok=True)
                
                degraded_dir = source_path / 'degraded'
                
                for f in files:
                    shutil.copy2(f, out_clean / f.name)
                    d_file = degraded_dir / f.name
                    if d_file.exists():
                        shutil.copy2(d_file, out_degraded / f.name)
            else:
                out_clean = split_out_dir / 'clean'
                out_clean.mkdir(parents=True, exist_ok=True)
                
                for f in files:
                    shutil.copy2(f, out_clean / f.name)
                    
            result_counts[split_name] = len(files)
            logger.info(f"Copied {len(files)} files to {split_out_dir}")
            
        return result_counts
