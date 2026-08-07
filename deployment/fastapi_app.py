import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import time
import logging
from typing import Optional, List
from pathlib import Path
import zipfile

import cv2
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.models.restoration_model import MultiTaskRestorationNet

app = FastAPI(
    title='SEM Image Restoration API',
    description='AI-based restoration of degraded semiconductor images',
    version='1.0.0'
)

# Add CORS
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# Global model and config
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = os.environ.get('MODEL_CHECKPOINT', 'weights/best_model.pth')

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.expand_dims(np.expand_dims(img_np, axis=0), axis=0)
    return torch.from_numpy(img_np).to(device)

def postprocess_image(tensor: torch.Tensor) -> bytes:
    tensor = tensor.cpu().detach().numpy()
    tensor = np.squeeze(tensor)
    tensor = np.clip(tensor, 0, 1) * 255.0
    img = Image.fromarray(tensor.astype(np.uint8), mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

@app.on_event('startup')
async def startup():
    global model
    if os.path.exists(CHECKPOINT_PATH):
        model = MultiTaskRestorationNet(in_channels=1, out_channels=1, base_channels=64)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        logging.info(f"Model loaded successfully from {CHECKPOINT_PATH} on {device}")
    else:
        logging.warning(f"Checkpoint not found at {CHECKPOINT_PATH}. API will not be fully functional.")

@app.get('/health')
async def health():
    return {'status': 'healthy', 'model_loaded': model is not None, 'device': device}

@app.post('/restore')
async def restore_image(file: UploadFile = File(...), task: Optional[str] = Query('all')):
    """Restore a single degraded SEM image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
        
    try:
        contents = await file.read()
        input_tensor = preprocess_image(contents)
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        result_bytes = postprocess_image(output_tensor)
        
        return StreamingResponse(io.BytesIO(result_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/batch')
async def batch_restore(files: List[UploadFile] = File(...)):
    """Restore multiple images."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
        
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                contents = await file.read()
                input_tensor = preprocess_image(contents)
                
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                    
                result_bytes = postprocess_image(output_tensor)
                zip_file.writestr(f"restored_{file.filename.split('.')[0]}.png", result_bytes)
                
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=restored_images.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/analyze')
async def analyze_image(file: UploadFile = File(...)):
    """Analyze image quality metrics (mockup since full IQA not detailed)."""
    return JSONResponse(content={
        "psnr_estimate": 28.5,
        "ssim_estimate": 0.85,
        "sharpness": 120.4,
        "noise_level": 15.2
    })

@app.get('/metrics')
async def model_metrics():
    """Return model performance metrics."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
        
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "model_type": "MultiTaskRestorationNet",
        "parameters": params,
        "device": device,
        "supported_formats": ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
