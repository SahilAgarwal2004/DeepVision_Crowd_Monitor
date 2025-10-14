import os
import glob
import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CrowdDataset(Dataset):
    """
    Fixed CrowdDataset using simple normalization (matches working code)
    No H5 files, no ImageNet normalization, no complex augmentation
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to train_data or test_data folder
            transform: IGNORED - kept for API compatibility
        """
        self.root_dir = root_dir
        self.image_size = (512, 512)  # Fixed size
        self.sigma = 4  # Gaussian sigma
        
        # Load image paths
        images_dir = os.path.join(root_dir, "images")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        self.image_paths = sorted(
            glob.glob(os.path.join(images_dir, "*.jpg")) +
            glob.glob(os.path.join(images_dir, "*.png"))
        )
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in {images_dir}")
        
        self.gt_dir = os.path.join(root_dir, "ground-truth")
        
        print(f"Found {len(self.image_paths)} images in {images_dir}")
        print(f"Ground truth directory: {self.gt_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and resize image
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        original_h, original_w = img.shape[:2]
        img = cv2.resize(img, self.image_size)
        
        # Load ground truth points
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(self.gt_dir, f"GT_{base_name}.mat")
        
        if not os.path.exists(gt_path):
            gt_path = os.path.join(self.gt_dir, f"{base_name}.mat")
        
        points = np.zeros((0, 2))
        if os.path.exists(gt_path):
            try:
                mat = sio.loadmat(gt_path)
                # Try different formats
                try:
                    points = np.array(mat["image_info"][0, 0][0, 0][0], dtype=np.float32)
                except:
                    try:
                        points = np.array(mat["image_info"][0, 0][0, 0][0, 0], dtype=np.float32)
                    except Exception as e:
                        print(f"[WARN] Could not parse {gt_path}: {e}")
            except Exception as e:
                print(f"[WARN] Could not load {gt_path}: {e}")
        
        # Scale points to resized image
        if points.size > 0:
            points[:, 0] = points[:, 0] * (self.image_size[1] / original_w)
            points[:, 1] = points[:, 1] * (self.image_size[0] / original_h)
        
        # Generate simple density map
        density_map = np.zeros(self.image_size, dtype=np.float32)
        
        for point in points:
            x = int(round(point[0]))
            y = int(round(point[1]))
            if 0 <= x < self.image_size[1] and 0 <= y < self.image_size[0]:
                density_map[y, x] += 1
        
        # Apply Gaussian blur (simple, fixed sigma)
        if density_map.sum() > 0:
            kernel_size = self.sigma * 4 + 1
            density_map = cv2.GaussianBlur(
                density_map,
                (kernel_size, kernel_size),
                self.sigma
            )
        
        # Image as tensor (0-1 range)
        img_tensor = torch.from_numpy(img.astype('float32') / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Density map as tensor 
        density_tensor = torch.from_numpy(density_map).float()
        
        return img_tensor, density_tensor


def get_dataloader(root_dir, batch_size=8, shuffle=True, num_workers=2):
    """
    Create dataloader with fixed preprocessing
    """
    dataset = CrowdDataset(root_dir, transform=None)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False  
    )