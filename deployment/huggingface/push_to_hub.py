import os
import sys
import argparse
import json
import logging
import tempfile
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model_card(model_name, metrics=None):
    """Create a model card for HuggingFace."""
    with open(os.path.join(os.path.dirname(__file__), 'model_card.md'), 'r') as f:
        template = f.read()
        
    template = template.replace("{{MODEL_NAME}}", model_name)
    
    metrics_str = ""
    if metrics:
        metrics_str = "| Metric | Value |\n|---|---|\n"
        for k, v in metrics.items():
            metrics_str += f"| {k} | {v} |\n"
    else:
        metrics_str = "Metrics not available."
        
    template = template.replace("{{METRICS}}", metrics_str)
    return template

def push_to_hub(repo_name, checkpoint_path, onnx_path=None, private=False, token=None):
    """Push model and artifacts to HuggingFace Hub."""
    logger.info(f"Pushing to HuggingFace Hub: {repo_name}")
    api = HfApi(token=token)
    
    try:
        repo_url = create_repo(repo_name, private=private, token=token, exist_ok=True)
        logger.info(f"Repository ready at {repo_url}")
    except Exception as e:
        logger.error(f"Failed to create repo: {e}")
        return None
        
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy model weights
        dest_ckpt = os.path.join(temp_dir, "pytorch_model.pth")
        shutil.copy(checkpoint_path, dest_ckpt)
        logger.info(f"Copied checkpoint to {dest_ckpt}")
        
        # Copy ONNX if provided
        if onnx_path and os.path.exists(onnx_path):
            dest_onnx = os.path.join(temp_dir, "model.onnx")
            shutil.copy(onnx_path, dest_onnx)
            logger.info(f"Copied ONNX to {dest_onnx}")
            
        # Create and write Model Card
        card_content = create_model_card(repo_name.split('/')[-1])
        with open(os.path.join(temp_dir, "README.md"), 'w') as f:
            f.write(card_content)
            
        # Create config
        config = {
            "model_type": "MultiTaskRestorationNet",
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 64
        }
        with open(os.path.join(temp_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=4)
            
        # Upload
        logger.info("Uploading folder...")
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=repo_name,
            repo_type="model",
            token=token
        )
        logger.info(f"Upload complete! Model available at: https://huggingface.co/{repo_name}")
        return repo_url

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-name', required=True, help="HF repo name (e.g., username/sem-restoration)")
    parser.add_argument('--checkpoint', required=True, help="Path to PyTorch checkpoint")
    parser.add_argument('--onnx-path', default=None, help="Path to ONNX model")
    parser.add_argument('--private', action='store_true', help="Make repo private")
    parser.add_argument('--token', default=None, help="HF access token")
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    
    push_to_hub(args.repo_name, args.checkpoint, args.onnx_path, args.private, token)

if __name__ == '__main__':
    main()
