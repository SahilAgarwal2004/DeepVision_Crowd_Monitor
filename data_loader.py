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

def create_h5_from_mat(img_path, gt_path):
    mat_path = gt_path.replace(".h5", ".mat")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Missing ground truth file: {mat_path}")

    mat = sio.loadmat(mat_path)
    points = mat["image_info"][0, 0][0, 0][0]

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    for x, y in points:
        if int(y) < h and int(x) < w:
            density[int(y), int(x)] = 1

    with h5py.File(gt_path, "w") as hf:
        hf["density"] = density

    return density


class CrowdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, "images", img)
            for img in os.listdir(os.path.join(root_dir, "images"))
            if img.endswith(".jpg")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        gt_path = img_path.replace("images", "ground_truth").replace(".jpg", ".h5")
        with h5py.File(gt_path, "r") as hf:
            target = np.asarray(hf["density"]).astype(np.float32)
        old_sum = target.sum()
        target_resized = cv2.resize(target, (512, 512), interpolation=cv2.INTER_CUBIC)
        resized_sum = target_resized.sum()
        if resized_sum > 0:
            target_resized *= (old_sum / resized_sum)
        target = target_resized
        target = torch.tensor(target, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, target


def get_dataloader(root_dir, batch_size=8, shuffle=True, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = CrowdDataset(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def visualize_sample(dataset, idx=0, save_path=None):
    image, density = dataset[idx]
    img_disp = image.permute(1, 2, 0).cpu().numpy()
    img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_disp)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(density, cmap="jet")
    plt.title(f"Sum={density.sum():.2f}")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    data_root = "Dataset/ShanghaiTech/part_A/train_data"
    dataloader = get_dataloader(data_root, batch_size=4)
    for images, targets in dataloader:
        print("Batch image shape:", images.shape)
        print("Batch target shape:", targets.shape)
        break
    dataset = CrowdDataset(data_root,
                           transform=transforms.Compose([
                               transforms.Resize((512, 512)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                           ]))
    visualize_sample(dataset, idx=0)