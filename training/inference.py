
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Model configuration
IMG_HEIGHT = 256
IMG_WIDTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CSRNet Model Definition
class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        
        vgg = models.vgg16(pretrained=load_weights)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(),
        )
        
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        
        if x.shape[2:] != (IMG_HEIGHT, IMG_WIDTH):
            x = F.interpolate(x, size=(IMG_HEIGHT, IMG_WIDTH), 
                            mode='bilinear', align_corners=False)
        return x


def load_model(model_path):
    """Load trained model"""
    model = CSRNet(load_weights=False).to(DEVICE)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # Handle different save formats
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()
    print(f"✅ Model loaded from: {model_path}")
    return model


def preprocess_image(image_path):
    """Preprocess image for inference"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize
    img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Simple normalization (matches training)
    img_array = np.array(img_resized).astype('float32') / 255.0
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    return img_tensor, img_resized, original_size


def predict(model, image_path):
    """Run inference on image"""
    img_tensor, img_resized, original_size = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(img_tensor)
        crowd_count = int(output.sum().item())
        density_map = output.squeeze().cpu().numpy()
    
    return {
        'count': crowd_count,
        'density_map': density_map,
        'image': img_resized,
        'original_size': original_size
    }


def visualize_results(results, save_path=None):
    """Visualize prediction results"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original image
    axes[0].imshow(results['image'])
    axes[0].set_title(f"Original Image\n{results['original_size'][0]}x{results['original_size'][1]}", 
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Density map
    im = axes[1].imshow(results['density_map'], cmap='jet', alpha=0.8)
    axes[1].set_title("Density Map", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(results['image'])
    axes[2].imshow(results['density_map'], cmap='jet', alpha=0.5)
    axes[2].set_title(f"Overlay\nPredicted Count: {results['count']}", 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='CSRNet Crowd Counting Inference')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='csrnet_best.pth',
                       help='Path to trained model')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization (optional)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CSRNet CROWD COUNTING INFERENCE")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print("-"*60)
    
    # Load model
    model = load_model(args.model)
    
    # Run inference
    print("\n🔍 Running inference...")
    results = predict(model, args.image)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Predicted Crowd Count: {results['count']}")
    print(f"Image Size: {results['original_size'][0]}x{results['original_size'][1]}")
    print(f"Density Map Range: [{results['density_map'].min():.4f}, {results['density_map'].max():.4f}]")
    print("="*60)
    
    # Visualize
    if not args.no_viz:
        print("\n📊 Generating visualization...")
        visualize_results(results, save_path=args.save)
    
    return results


if __name__ == "__main__":
    main()






