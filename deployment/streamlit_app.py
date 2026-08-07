import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import io
import time

from src.models.restoration_model import MultiTaskRestorationNet

st.set_page_config(page_title='SEM Image Restoration', page_icon='🔬', layout='wide')

@st.cache_resource
def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        return None
    model = MultiTaskRestorationNet(in_channels=1, out_channels=1, base_channels=64)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image: Image.Image, device: str) -> torch.Tensor:
    img = image.convert('L') # to grayscale
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.expand_dims(np.expand_dims(img_np, axis=0), axis=0)
    return torch.from_numpy(img_np).to(device)

def postprocess_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.cpu().detach().numpy()
    tensor = np.squeeze(tensor)
    tensor = np.clip(tensor, 0, 1) * 255.0
    return Image.fromarray(tensor.astype(np.uint8), mode='L')

def main():
    st.title('🔬 AI-Based Semiconductor Image Restoration')
    st.markdown('Upload a degraded SEM image to restore it.')
    
    st.sidebar.title("Settings")
    checkpoint_path = st.sidebar.text_input("Checkpoint Path", value="weights/best_model.pth")
    device_option = st.sidebar.selectbox("Device", ["cpu", "cuda"])
    
    device = 'cuda' if device_option == 'cuda' and torch.cuda.is_available() else 'cpu'
    
    model = load_model(checkpoint_path, device)
    if model is None:
        st.sidebar.error(f"Could not load model from {checkpoint_path}. Please check path.")
    else:
        st.sidebar.success(f"Model loaded successfully on {device}")
    
    uploaded_file = st.file_uploader('Upload SEM Image', type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'])
    
    if uploaded_file and model is not None:
        col1, col2 = st.columns(2)
        
        orig_img = Image.open(uploaded_file)
        with col1:
            st.subheader("Original Image")
            st.image(orig_img, use_column_width=True)
            
        with st.spinner("Restoring image..."):
            start_time = time.time()
            input_tensor = preprocess_image(orig_img, device)
            
            with torch.no_grad():
                output_tensor = model(input_tensor)
                
            restored_img = postprocess_image(output_tensor['output'] if isinstance(output_tensor, dict) else output_tensor)
            end_time = time.time()
            
        with col2:
            st.subheader("Restored Image")
            st.image(restored_img, use_column_width=True)
            
        st.success(f"Restoration completed in {end_time - start_time:.2f} seconds.")
        
        # Download button
        buf = io.BytesIO()
        restored_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="Download Restored Image",
            data=byte_im,
            file_name="restored.png",
            mime="image/png"
        )
        
        # Difference Map
        orig_gray = np.array(orig_img.convert('L'))
        rest_gray = np.array(restored_img)
        diff = cv2.absdiff(orig_gray, rest_gray)
        diff_heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        st.subheader("Difference Map")
        st.image(cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB), use_column_width=True)

if __name__ == '__main__':
    main()
