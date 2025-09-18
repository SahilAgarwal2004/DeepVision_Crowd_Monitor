import os
import cv2
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree


def generate_density_map(img, points, k=3):
    h, w = img.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    points = np.array(points)
    if len(points.shape) == 1:
        points = points.reshape(1, -1)
    
    # Filter valid points
    valid_points = []
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            valid_points.append([x, y])
    
    if len(valid_points) == 0:
        return density
        
    valid_points = np.array(valid_points)
    
    # Calculate adaptive sigma
    if len(valid_points) > k:
        tree = KDTree(valid_points, leafsize=2048)
        distances, _ = tree.query(valid_points, k=k+1)
    
    # Create density map
    for i, pt in enumerate(valid_points):
        x, y = int(pt[0]), int(pt[1])
        
        if len(valid_points) > k:
            sigma = max(2.0, min(np.mean(distances[i][1:]) * 0.3, 15.0))
        else:
            sigma = max(3.0, min(h, w) / 20.0)
        
        temp_density = np.zeros((h, w), dtype=np.float32)
        temp_density[y, x] = 1.0
        temp_density = gaussian_filter(temp_density, sigma=sigma, mode="constant")
        density += temp_density

    return density


def create_h5_from_mat(img_path, h5_path):

    img_name = os.path.basename(img_path).replace(".jpg", "")
    
    # Try different possible locations for MAT files
    possible_paths = [
        os.path.join(os.path.dirname(h5_path), f"GT_{img_name}.mat"),
        os.path.join(os.path.dirname(os.path.dirname(h5_path)), "ground_truth", f"GT_{img_name}.mat"),
        os.path.join(os.path.dirname(img_path).replace("images", "ground_truth"), f"GT_{img_name}.mat")
    ]
    
    mat_path = None
    for path in possible_paths:
        if os.path.exists(path):
            mat_path = path
            break
    
    if mat_path is None:
        raise FileNotFoundError(f"Ground truth file not found for {img_name}")

    try:
        mat = sio.loadmat(mat_path)
        points = mat["image_info"][0, 0][0, 0][0]
        
        if len(points) == 0:
            print(f"Warning: No points found in {mat_path}")
            
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        density = generate_density_map(img, points)
        
        with h5py.File(h5_path, "w") as hf:
            hf["density"] = density

        return density
    except Exception as e:
        raise RuntimeError(f"Error processing {img_name}: {str(e)}")


class CrowdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        images_dir = os.path.join(root_dir, "images")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
            
        self.image_paths = [
            os.path.join(images_dir, img)
            for img in os.listdir(images_dir)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in {images_dir}")
            
        self.transform = transform
        print(f"Found {len(self.image_paths)} images in dataset")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        img_name = os.path.basename(img_path).replace(".jpg", "")
        h5_path = os.path.join(self.root_dir, "ground-truth", f"{img_name}.h5")

        if not os.path.exists(h5_path):
            target = create_h5_from_mat(img_path, h5_path)
        else:
            with h5py.File(h5_path, "r") as hf:
                target = np.asarray(hf["density"]).astype(np.float32)


        old_sum = target.sum()
        target_resized = cv2.resize(target, (512, 512), interpolation=cv2.INTER_CUBIC)
        if target_resized.sum() > 0:
            target_resized *= (old_sum / target_resized.sum())
        
        target = torch.tensor(target_resized, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target


def get_dataloader(root_dir, batch_size=8, shuffle=True, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = CrowdDataset(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def visualize_sample(dataset, idx=0):
    image, density = dataset[idx]
    
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_disp = (image * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()
    
    # Process density map
    density_np = density.numpy()
    if density_np.max() > 0:
        density_vis = np.power(density_np / density_np.max(), 0.4)
    else:
        density_vis = density_np
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_disp)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Density map
    im = axes[1].imshow(density_vis, cmap='hot', interpolation='bilinear')
    axes[1].set_title(f'Density Map (Count: {density_np.sum():.1f})')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], shrink=0.7)
    
    # Overlay
    axes[2].imshow(img_disp)
    axes[2].imshow(density_vis, cmap='jet', alpha=0.6, interpolation='bilinear')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_root = "Dataset/ShanghaiTech/part_A/train_data"
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = CrowdDataset(data_root, transform=transform)
    
    # Visualize sample only
    visualize_sample(dataset, idx=0)