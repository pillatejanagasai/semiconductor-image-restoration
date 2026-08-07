import os
import torch
import pytest
from PIL import Image
from pathlib import Path
from src.datasets.dataset import SEMDataset, DataSplitter

@pytest.fixture
def temp_images(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        img = Image.new('L', (128, 128), color = i * 10)
        img.save(img_dir / f"img_{i}.png")
    return img_dir

@pytest.fixture
def temp_paired_images(tmp_path):
    hq_dir = tmp_path / "hq"
    lq_dir = tmp_path / "lq"
    hq_dir.mkdir()
    lq_dir.mkdir()
    for i in range(5):
        img_hq = Image.new('L', (128, 128), color = i * 10)
        img_lq = Image.new('L', (128, 128), color = i * 5)
        img_hq.save(hq_dir / f"img_{i}.png")
        img_lq.save(lq_dir / f"img_{i}.png")
    return hq_dir, lq_dir

def test_sem_dataset_synthetic_mode(temp_images):
    dataset = SEMDataset(image_dir=temp_images, mode='synthetic', patch_size=64)
    assert len(dataset) == 5
    sample = dataset[0]
    assert 'hq' in sample
    assert 'lq' in sample
    assert sample['hq'].shape == (1, 64, 64)
    assert sample['lq'].shape == (1, 64, 64)

def test_sem_dataset_paired_mode(temp_paired_images):
    hq_dir, lq_dir = temp_paired_images
    dataset = SEMDataset(image_dir=hq_dir, target_dir=lq_dir, mode='paired', patch_size=64)
    assert len(dataset) == 5
    sample = dataset[0]
    assert sample['hq'].shape == (1, 64, 64)
    assert sample['lq'].shape == (1, 64, 64)

def test_dataset_length(temp_images):
    dataset = SEMDataset(image_dir=temp_images, mode='synthetic')
    assert len(dataset) == 5

def test_dataset_output_format(temp_images):
    dataset = SEMDataset(image_dir=temp_images, mode='synthetic')
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert 'hq' in sample
    assert 'lq' in sample
    assert 'filename' in sample

def test_data_splitter(temp_images):
    files = list(temp_images.glob("*.png"))
    splitter = DataSplitter(files, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    train, val, test = splitter.split()
    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 1

def test_supported_formats(tmp_path):
    img_dir = tmp_path / "formats"
    img_dir.mkdir()
    Image.new('L', (128, 128)).save(img_dir / "img1.png")
    Image.new('L', (128, 128)).save(img_dir / "img2.jpg")
    Image.new('L', (128, 128)).save(img_dir / "img3.jpeg")
    Image.new('L', (128, 128)).save(img_dir / "img4.bmp")
    dataset = SEMDataset(image_dir=img_dir, mode='synthetic')
    assert len(dataset) == 4
