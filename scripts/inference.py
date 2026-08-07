import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Optional, List, Union, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.restoration_model import MultiTaskRestorationNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferenceEngine:
    """Production inference engine for semiconductor image restoration."""
    
    def __init__(self, checkpoint_path=None, onnx_path=None, device='auto', tile_size=256, tile_overlap=32):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = None
        self.ort_session = None
        
        if onnx_path and os.path.exists(onnx_path):
            self._load_onnx_model(onnx_path)
        elif checkpoint_path and os.path.exists(checkpoint_path):
            self._load_pytorch_model(checkpoint_path)
        else:
            logger.warning("No valid model path provided. Engine initialized without model.")

    def _load_pytorch_model(self, checkpoint_path):
        """Load PyTorch model from checkpoint."""
        logger.info(f"Loading PyTorch model from {checkpoint_path} to {self.device}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Instantiate model
        self.model = MultiTaskRestorationNet(in_channels=1, out_channels=1, base_channels=64)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()

    def _load_onnx_model(self, onnx_path):
        """Load ONNX model for inference."""
        logger.info(f"Loading ONNX model from {onnx_path}")
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
            self.device = 'onnx'
        except ImportError:
            logger.error("onnxruntime is not installed. Please install it to use ONNX models.")
            raise

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess input image."""
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=2)
            
        # Normalize to [0,1] float32
        image = image.astype(np.float32) / 255.0
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dim
        image = np.expand_dims(image, axis=0)
        
        return torch.from_numpy(image).to(self.device) if self.device != 'onnx' else image

    def postprocess(self, output) -> np.ndarray:
        """Postprocess model output."""
        if isinstance(output, torch.Tensor):
            output = output.cpu().detach().numpy()
            
        # Remove batch dim
        output = np.squeeze(output, axis=0)
        
        # CHW to HWC
        output = np.transpose(output, (1, 2, 0))
        
        # Remove channel dim if 1
        if len(output.shape) == 3 and output.shape[2] == 1:
            output = np.squeeze(output, axis=2)
            
        # Clamp to [0,1]
        output = np.clip(output, 0.0, 1.0)
        
        # Convert to uint8 [0,255]
        output = (output * 255.0).astype(np.uint8)
        
        return output

    def _infer_forward(self, x):
        if self.device == 'onnx':
            ort_inputs = {self.ort_session.get_inputs()[0].name: x}
            ort_outs = self.ort_session.run(None, ort_inputs)
            return ort_outs[0]
        else:
            with torch.no_grad():
                out = self.model(x)
                return out['output'] if isinstance(out, dict) else out

    def _tiled_inference(self, image_tensor):
        """Perform tiled inference for large images."""
        if self.device != 'onnx':
            b, c, h, w = image_tensor.shape
            device = image_tensor.device
            tensor_np = image_tensor.cpu().numpy()
        else:
            b, c, h, w = image_tensor.shape
            tensor_np = image_tensor

        stride = self.tile_size - self.tile_overlap
        
        # Calculate grid size
        h_idx = list(range(0, h - self.tile_size + 1, stride)) + ([h - self.tile_size] if h % stride != 0 else [])
        w_idx = list(range(0, w - self.tile_size + 1, stride)) + ([w - self.tile_size] if w % stride != 0 else [])
        
        if not h_idx: h_idx = [0]
        if not w_idx: w_idx = [0]

        out_tensor = np.zeros((b, c, h, w), dtype=np.float32)
        weight_map = np.zeros((b, c, h, w), dtype=np.float32)
        
        # Bartlett window for blending
        window_1d = np.bartlett(self.tile_size)
        window_2d = np.outer(window_1d, window_1d)
        window_2d = np.expand_dims(np.expand_dims(window_2d, axis=0), axis=0)
        
        for y in h_idx:
            for x in w_idx:
                tile = tensor_np[:, :, y:y+self.tile_size, x:x+self.tile_size]
                
                # Pad tile if it's smaller than tile_size
                pad_y = self.tile_size - tile.shape[2]
                pad_x = self.tile_size - tile.shape[3]
                if pad_y > 0 or pad_x > 0:
                    tile = np.pad(tile, ((0,0), (0,0), (0,pad_y), (0,pad_x)), mode='reflect')
                
                if self.device != 'onnx':
                    tile_tensor = torch.from_numpy(tile).to(device)
                    out_tile = self._infer_forward(tile_tensor).cpu().numpy()
                else:
                    out_tile = self._infer_forward(tile)
                
                # Unpad
                if pad_y > 0 or pad_x > 0:
                    out_tile = out_tile[:, :, :self.tile_size-pad_y, :self.tile_size-pad_x]
                
                win = window_2d[:, :, :out_tile.shape[2], :out_tile.shape[3]]
                out_tensor[:, :, y:y+out_tile.shape[2], x:x+out_tile.shape[3]] += out_tile * win
                weight_map[:, :, y:y+out_tile.shape[2], x:x+out_tile.shape[3]] += win
                
        weight_map[weight_map == 0] = 1e-6
        out_tensor /= weight_map
        
        if self.device != 'onnx':
            return torch.from_numpy(out_tensor).to(device)
        return out_tensor

    def infer_single(self, image_path, output_path=None):
        """Restore a single image."""
        start_time = time.time()
        # Load image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Could not read image {image_path}")
            return None
            
        # Preprocess
        tensor = self.preprocess(img)
        
        # Inference
        h, w = tensor.shape[2:] if self.device != 'onnx' else tensor.shape[2:]
        if h > self.tile_size or w > self.tile_size:
            out_tensor = self._tiled_inference(tensor)
        else:
            out_tensor = self._infer_forward(tensor)
            
        # Postprocess
        restored = self.postprocess(out_tensor)
        
        infer_time = time.time() - start_time
        logger.info(f"Inference complete in {infer_time:.3f}s")
        
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(str(output_path), restored)
            
        return restored

    def infer_batch(self, image_paths, output_dir):
        """Process a list of images."""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        for i, path in enumerate(image_paths):
            logger.info(f"Processing [{i+1}/{len(image_paths)}]: {path}")
            out_path = os.path.join(output_dir, os.path.basename(path))
            res = self.infer_single(path, out_path)
            results.append((path, out_path))
        return results

    def infer_folder(self, input_dir, output_dir, recursive=True):
        """Process all images in a folder."""
        supported_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        image_paths = []
        
        path_obj = Path(input_dir)
        glob_pattern = '**/*' if recursive else '*'
        
        for p in path_obj.glob(glob_pattern):
            if p.is_file() and p.suffix.lower() in supported_exts:
                image_paths.append(str(p))
                
        if not image_paths:
            logger.warning(f"No supported images found in {input_dir}")
            return
            
        logger.info(f"Found {len(image_paths)} images to process")
        os.makedirs(output_dir, exist_ok=True)
        
        for i, path in enumerate(image_paths):
            rel_path = os.path.relpath(path, input_dir)
            out_path = os.path.join(output_dir, rel_path)
            
            logger.info(f"Processing [{i+1}/{len(image_paths)}]: {rel_path}")
            self.infer_single(path, out_path)

    @staticmethod
    def export_onnx(checkpoint_path, onnx_path, input_size=(1,1,256,256), dynamic_axes=True):
        """Export PyTorch model to ONNX format."""
        logger.info(f"Exporting PyTorch model {checkpoint_path} to ONNX {onnx_path}")
        device = 'cpu'
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = MultiTaskRestorationNet(in_channels=1, out_channels=1, base_channels=64)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        
        dummy_input = torch.randn(*input_size, device=device)
        
        dynamic = None
        if dynamic_axes:
            dynamic = {
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
            
        os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic
        )
        
        logger.info(f"ONNX export successful: {onnx_path}")
        
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model checker passed")
        except ImportError:
            logger.warning("onnx package not installed, skipping checker")

def main():
    parser = argparse.ArgumentParser(description='Semiconductor Image Restoration Inference')
    parser.add_argument('--mode', choices=['single', 'batch', 'folder', 'export_onnx'], required=True)
    parser.add_argument('--input', type=str, help='Input image path or directory')
    parser.add_argument('--output', type=str, default='outputs', help='Output path or directory')
    parser.add_argument('--checkpoint', type=str, default='weights/best_model.pth')
    parser.add_argument('--onnx-path', type=str, default='weights/model.onnx')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--tile-overlap', type=int, default=32)
    args = parser.parse_args()
    
    if args.mode == 'export_onnx':
        InferenceEngine.export_onnx(args.checkpoint, args.onnx_path)
        return
        
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx_path if args.onnx_path and os.path.exists(args.onnx_path) else None,
        device=args.device,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap
    )
    
    if args.mode == 'single':
        if not args.input:
            raise ValueError("--input is required for single mode")
        engine.infer_single(args.input, args.output)
    elif args.mode == 'batch':
        if not args.input:
            raise ValueError("--input is required for batch mode (comma-separated paths)")
        paths = args.input.split(',')
        engine.infer_batch(paths, args.output)
    elif args.mode == 'folder':
        if not args.input:
            raise ValueError("--input is required for folder mode")
        engine.infer_folder(args.input, args.output)

if __name__ == '__main__':
    main()
