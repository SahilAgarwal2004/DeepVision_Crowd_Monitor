import os
import sys
import yaml
import torch
import wandb
from torchvision import transforms

from src.model.csrnet import CSRNet
from src.data.data_loader import CrowdDataset
from src.training.train import CrowdTrainer


def load_config(path='config.yaml'):
    """Load YAML configuration file"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print("DeepVision Crowd Monitor - Training")
    print("=" * 50)
    
    # Load config
    try:
        config = load_config()
        print("Config loaded")
    except FileNotFoundError:
        print("config.yaml not found in main folder!")
        return
    
    # Check dataset
    data_path = config['data']['root_dir']
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        print("Update the path in config.yaml")
        return
    
    print(f"Dataset found: {data_path}")
    
    # Initialize WandB
    wandb.init(
        project=config['logging']['wandb']['project'],
        entity=config['logging']['wandb']['entity'],
        config=config,
        name="csrnet_training"
    )
    print(" WandB initialized")
    
    # Create model
    model = CSRNet(pretrained=config['model']['pretrained'])
    print(f"CSRNet model created")
    
    # Create trainer
    trainer = CrowdTrainer(model, config)
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize(config['data']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Setup dataset
    trainer.setup_data(CrowdDataset, transform)
    
    # Start training
    try:
        print("\n Starting Training...")
        print("=" * 50)
        best_mae = trainer.train()
        
        print("\n Training Complete!")
        print(f" Best MAE: {best_mae:.2f}")
        wandb.finish()
        
    except KeyboardInterrupt:
        print("\n Training stopped by user")
        wandb.finish()
    except Exception as e:
        print(f"\n Training failed: {e}")
        wandb.finish()
        raise


if __name__ == "__main__":
    main()