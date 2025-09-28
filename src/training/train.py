import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import wandb
from tqdm import tqdm
from datetime import datetime


class CrowdCountingLoss(nn.Module):
    """Combined MSE loss for pixels and counts"""
    def __init__(self, count_weight=0.01):
        super().__init__()
        self.mse = nn.MSELoss()
        self.count_weight = count_weight
    
    def forward(self, pred, target):
        # Resize if needed
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)
        
        # Pixel loss + count loss
        pixel_loss = self.mse(pred, target)
        count_loss = self.mse(pred.sum(dim=[2,3]), target.sum(dim=[2,3]))
        
        return pixel_loss + self.count_weight * count_loss


def calculate_mae(pred, target):
    """Calculate Mean Absolute Error in count"""
    pred_count = pred.sum(dim=[2,3]).detach().cpu().numpy()
    target_count = target.sum(dim=[2,3]).detach().cpu().numpy()
    return np.mean(np.abs(pred_count - target_count))


class CrowdTrainer:
    def __init__(self, model, config):
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup training components
        self.criterion = CrowdCountingLoss()
        param_groups = model.get_parameter_groups(
            lr_frontend=config['training']['learning_rate'] * 0.1,
            lr_backend=config['training']['learning_rate']
        )
        self.optimizer = optim.Adam(param_groups, weight_decay=config['training']['weight_decay'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        
        # Results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(config['logging']['results_dir'], f"csrnet_{timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.best_mae = float('inf')
        print(f"🚀 Trainer ready | Device: {self.device} | Params: {sum(p.numel() for p in model.parameters()):,}")
    
    def setup_data(self, dataset_class, transform):
        """Split dataset and create loaders"""
        full_dataset = dataset_class(self.config['data']['root_dir'], transform=transform)
        
        # Train/val split
        train_size = int(self.config['training']['train_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        
        # Data loaders
        from torch.utils.data import DataLoader
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config['training']['batch_size'],
            shuffle=True, num_workers=self.config['data']['num_workers'], pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.config['training']['batch_size'],
            shuffle=False, num_workers=self.config['data']['num_workers'], pin_memory=True
        )
        
        print(f"📊 Dataset: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
    
    def train_epoch(self, epoch):
        """Single training epoch"""
        self.model.train()
        total_loss, total_mae, count = 0, 0, 0
        
        pbar = tqdm(self.train_loader, desc=f'Train {epoch+1}/{self.config["training"]["num_epochs"]}')
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.unsqueeze(1).to(self.device)
            
            # Forward + backward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            mae = calculate_mae(outputs, targets)
            total_loss += loss.item()
            total_mae += mae
            count += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'MAE': f'{mae:.1f}'})
            
            # Log to wandb
            if batch_idx % self.config['logging']['wandb']['log_interval'] == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/mae': mae,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch + 1
                })
        
        return total_loss / count, total_mae / count
    
    def validate_epoch(self, epoch):
        """Single validation epoch"""
        self.model.eval()
        total_loss, total_mae, count = 0, 0, 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(self.val_loader, desc=f'Val {epoch+1}')):
                images, targets = images.to(self.device), targets.unsqueeze(1).to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                mae = calculate_mae(outputs, targets)
                
                total_loss += loss.item()
                total_mae += mae
                count += 1
                
                # Log visual examples every 5 epochs (only first batch)
                if batch_idx == 0 and epoch % 5 == 0:
                    # Denormalize image for display
                    img = images[0].detach().cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_denorm = (img * std + mean).clamp(0, 1)
                    
                    # Get density maps
                    pred_map = outputs[0, 0].detach().cpu()  # Remove batch and channel dims
                    gt_map = targets[0, 0].detach().cpu()
                    
                    # Normalize for better visualization
                    pred_vis = pred_map / (pred_map.max() + 1e-8)
                    gt_vis = gt_map / (gt_map.max() + 1e-8)
                    
                    wandb.log({
                        "examples": [
                            wandb.Image(img_denorm, caption="Input Image"),
                            wandb.Image(pred_vis, caption=f"Predicted (Count: {pred_map.sum():.1f})"),
                            wandb.Image(gt_vis, caption=f"Ground Truth (Count: {gt_map.sum():.1f})")
                        ],
                        "epoch": epoch + 1
                    })
        
        return total_loss / count, total_mae / count
    
    def save_model(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mae': self.best_mae,
            'config': self.config
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.results_dir, 'best_model.pth'))
            print(f"New best model! MAE: {self.best_mae:.2f}")
    
    def train(self):
        """Main training loop"""
        print(f"Training for {self.config['training']['num_epochs']} epochs")
        start_time = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            # Train & validate
            train_loss, train_mae = self.train_epoch(epoch)
            val_loss, val_mae = self.validate_epoch(epoch)
            
            self.scheduler.step()
            
            # Save best model
            if val_mae < self.best_mae:
                self.best_mae = val_mae
                self.save_model(epoch, is_best=True)
            
            # Log to wandb
            wandb.log({
                'train/epoch_loss': train_loss,
                'train/epoch_mae': train_mae,
                'val/epoch_loss': val_loss,
                'val/epoch_mae': val_mae,
                'best_mae': self.best_mae,
                'epoch': epoch + 1
            })
            
            # Print results
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f}/{train_mae:.1f} | "
                  f"Val: {val_loss:.4f}/{val_mae:.1f} | Best: {self.best_mae:.1f}")
        
        total_time = time.time() - start_time
        print(f"Training done! Best MAE: {self.best_mae:.2f} in {total_time/3600:.1f}h")
        
        return self.best_mae