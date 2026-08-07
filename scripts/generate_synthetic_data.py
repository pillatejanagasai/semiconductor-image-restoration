import os
import sys
import cv2
import numpy as np
import random
import argparse
from tqdm import tqdm

def speckle_noise(img, mean=0, std=0.2):
    noise = np.random.normal(mean, std, img.shape)
    noisy_img = img.astype(np.float32)
    noisy_img = noisy_img + noisy_img * noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def generate_grid(size=512):
    img = np.zeros((size, size), dtype=np.uint8)
    spacing = random.randint(20, 80)
    thickness = random.randint(1, 5)
    
    # Draw vertical lines
    for x in range(0, size, spacing):
        cv2.line(img, (x, 0), (x, size), 255, thickness)
        
    # Draw horizontal lines
    for y in range(0, size, spacing):
        cv2.line(img, (0, y), (size, y), 255, thickness)
        
    return img

def generate_traces(size=512):
    img = np.zeros((size, size), dtype=np.uint8)
    num_traces = random.randint(5, 20)
    for _ in range(num_traces):
        thickness = random.randint(2, 10)
        y = random.randint(0, size)
        cv2.line(img, (0, y), (size, y), 255, thickness)
        
        # sometimes add a bend
        if random.random() > 0.5:
            x_bend = random.randint(100, size-100)
            y_bend = random.randint(0, size)
            cv2.line(img, (x_bend, y), (size, y_bend), 255, thickness)
            
    # randomly rotate
    angle = random.randint(0, 180)
    M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
    img = cv2.warpAffine(img, M, (size, size))
    return img

def generate_dendrites(size=512):
    """Matches Figure 2 dendrite sample from Hackathon PDF"""
    img = np.zeros((size, size), dtype=np.uint8)
    num_stars = random.randint(2, 8)
    for _ in range(num_stars):
        cx, cy = random.randint(50, size-50), random.randint(50, size-50)
        num_spokes = random.randint(10, 40)
        max_length = random.randint(50, 200)
        for _ in range(num_spokes):
            angle = random.uniform(0, 2 * np.pi)
            length = random.uniform(max_length * 0.3, max_length)
            ex = int(cx + length * np.cos(angle))
            ey = int(cy + length * np.sin(angle))
            thickness = random.randint(1, 3)
            # draw with decreasing intensity
            color = random.randint(100, 255)
            cv2.line(img, (cx, cy), (ex, ey), color, thickness)
    return img

def generate_texture(size=512):
    """Matches Figure 1 texture sample from Hackathon PDF"""
    noise = np.random.randint(0, 256, (size//4, size//4), dtype=np.uint8)
    noise = cv2.resize(noise, (size, size), interpolation=cv2.INTER_LINEAR)
    # Add some structural polygons
    for _ in range(3):
        pts = np.random.randint(0, size, (4, 2), dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(noise, [pts], random.randint(100, 255))
    
    # Blur to make it look like SEM texture
    noise = cv2.GaussianBlur(noise, (15, 15), 0)
    return noise

def create_degraded_pair(gt_img):
    """Applies downsampling and speckle noise to create NoisyLR."""
    # 512x512 -> 256x256 (Resolution Reduction from Slide 9)
    lr_size = gt_img.shape[0] // 2
    lr_img = cv2.resize(gt_img, (lr_size, lr_size), interpolation=cv2.INTER_AREA)
    
    # Apply Speckle Noise
    std = random.uniform(0.1, 0.4)
    noisy_lr = speckle_noise(lr_img, std=std)
    
    # Optionally apply a little blur to mimic Gaussian optical degradation
    if random.random() > 0.5:
        noisy_lr = cv2.GaussianBlur(noisy_lr, (3, 3), 0)
        
    return noisy_lr

def main():
    parser = argparse.ArgumentParser(description="Procedural Synthetic Data Generator")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of image pairs to generate")
    parser.add_argument("--out_dir", type=str, default="dataset/synthetic", help="Output directory")
    args = parser.parse_args()
    
    gt_dir = os.path.join(args.out_dir, "clean")
    lr_dir = os.path.join(args.out_dir, "degraded")
    
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)
    
    generators = [generate_grid, generate_traces, generate_dendrites, generate_texture]
    
    print(f"Generating {args.num_samples} synthetic image pairs...")
    for i in tqdm(range(args.num_samples)):
        generator = random.choice(generators)
        gt_img = generator(size=512)
        noisy_lr = create_degraded_pair(gt_img)
        
        gt_path = os.path.join(gt_dir, f"synth_{i:05d}.png")
        lr_path = os.path.join(lr_dir, f"synth_{i:05d}.png")
        
        cv2.imwrite(gt_path, gt_img)
        cv2.imwrite(lr_path, noisy_lr)
        
    print(f"Success! {args.num_samples} paired images saved to {args.out_dir}")

if __name__ == "__main__":
    main()
