import os
import sys
import yaml
import torch
import wandb
from dotenv import load_dotenv

load_dotenv()

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
        print(" Config loaded")
        print(f"  Learning rate: {config['training']['learning_rate']}")
    except FileNotFoundError:
        print(" config.yaml not found!")
        return
    
    # Check dataset
    data_path = config['data']['root_dir']
    if not os.path.exists(data_path):
        print(f" Dataset not found: {data_path}")
        return
    
    print(f"✓ Dataset found: {data_path}")

    # Initialize WandB
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    wandb.init(
        project=config['logging']['wandb']['project'],
        entity=config['logging']['wandb']['entity'],
        config=config,
        name="csrnet_training_fixed"
    )
    print(" WandB initialized")
    
    # Create model
    model = CSRNet(pretrained=config['model']['pretrained'])
    print(f" CSRNet model created")
    
    # Create trainer
    trainer = CrowdTrainer(model, config)
    
    
    train_transform = None
    val_transform = None
    
    # Setup dataset
    trainer.setup_data(CrowdDataset, train_transform, val_transform)
    
    # Start training
    try:
        print("\n Starting Training...")
        print("=" * 50)
        best_mae = trainer.train()
        
        print("\n Training Complete!")
        print(f"  Best MAE: {best_mae:.2f}")
        wandb.finish()
        
    except KeyboardInterrupt:
        print("\n  Training stopped by user")
        wandb.finish()
    except Exception as e:
        print(f"\n Training failed: {e}")
        wandb.finish()
        raise


if __name__ == "__main__":
    main()