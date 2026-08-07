import os
import sys
import argparse
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.restoration_model import MultiTaskRestorationNet

class FastSEMTestDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        supported_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        self.image_paths = sorted([p for p in self.image_dir.glob('*') if p.is_file() and p.suffix.lower() in supported_exts])
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for TIFFs
            import tifffile
            img = tifffile.imread(str(path))
            if len(img.shape) == 3:
                if img.shape[2] == 3 or img.shape[2] == 4:
                    if img.dtype == np.uint8:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if img.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
                    else:
                        img = img.mean(axis=2)
                elif img.shape[0] == 3 or img.shape[0] == 4:
                    img = img.mean(axis=0)

        img_np = img.astype(np.float32) / 255.0
        img_np = np.expand_dims(img_np, axis=0) # [1, H, W]
        return torch.from_numpy(img_np), str(path.name)

def main():
    parser = argparse.ArgumentParser(description="SEMICON Image Restoration Submission")
    parser.add_argument("input_dir", type=str, help="Path to test images directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    # Load Model
    model = MultiTaskRestorationNet(in_channels=1, out_channels=1)
    checkpoint_path = 'weights/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Ensure it is present during evaluation.")
    
    model.to(device)
    model.eval()
    
    # DataLoader (Optimized for H100 GPU speed)
    dataset = FastSEMTestDataset(args.input_dir)
    # H100 can handle large batch sizes easily
    loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True, shuffle=False)
    
    with torch.no_grad():
        for batch_imgs, batch_names in loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            
            # Mixed precision (FP16) on Tensor Cores
            if torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(batch_imgs)
            else:
                outputs = model(batch_imgs)
                
            preds = outputs['output'] if isinstance(outputs, dict) else outputs
                
            preds = preds.cpu().float().numpy()
            
            # Save batch (I/O can be further optimized via multiprocessing if disk bound)
            for i, name in enumerate(batch_names):
                pred_img = preds[i].squeeze(0) # [H, W]
                pred_img = np.clip(pred_img, 0, 1) * 255.0
                pred_img = pred_img.astype(np.uint8)
                out_path = os.path.join(args.output_dir, name)
                cv2.imwrite(out_path, pred_img)

if __name__ == "__main__":
    main()
