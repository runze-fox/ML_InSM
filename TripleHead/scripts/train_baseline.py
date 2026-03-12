# -*- coding: utf-8 -*-
"""
Baseline Training Script
Basic training script to verify model architecture correctness (without using Optuna)
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys
import copy

# Add path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_train_val_datasets
from models.triple_head_model import TripleHeadPEModel
from models.losses import ComponentWeightedHAZLoss


class EarlyStopping:
    """
    Early stopping mechanism: monitors validation loss, stops training if no improvement for 'patience' epochs
    """
    def __init__(self, patience=50, min_delta=0, verbose=True):
        """
        Args:
            patience: Number of epochs to tolerate no improvement
            min_delta: Minimum change to qualify as an improvement
            verbose: Whether to print information
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  [EarlyStopping] Validation loss improved to {val_loss:.6f}")
        else:
            # No improvement
            self.counter += 1
            if self.verbose and self.counter % 10 == 0:
                print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n[EarlyStopping] Early stopping triggered! Best epoch: {self.best_epoch}, Best loss: {self.best_loss:.6f}")


def train_baseline(epochs=1000,
                   lr=0.001,
                   weight_decay=1e-5,
                   haz_multiplier=10.0,
                   base_channels=128,
                   alpha=1.0, beta=1.0, gamma=1.0,
                   data_dir=r'D:\CNN_InSM_New\data_input\numpy_data\from_interpo\Web15mm',
                   save_dir=r'D:\CNN_InSM_New\TripleHead\results\Web15mm'):
    """
    Baseline training process
    
    Args:
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        haz_multiplier: HAZ region weight multiplier
        base_channels: Model channels
        alpha, beta, gamma: Loss weights for PE11, PE22, PE33
        data_dir: Data directory
        save_dir: Model save path
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. Prepare datasets
    print("\n[Data] Preparing datasets...")
    train_dataset, valid_dataset = get_train_val_datasets(data_dir, valid_ratio=0.2)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    # 2. Initialize model
    print(f"\n[Model] Initializing TripleHeadPEModel (base_channels={base_channels})")
    model = TripleHeadPEModel(base_channels=base_channels).to(device)
    print(f"  Total parameters: {model.get_num_parameters():,}")
    
    # 3. Loss function and optimizer
    criterion = ComponentWeightedHAZLoss(
        haz_multiplier=haz_multiplier,
        alpha=alpha, beta=beta, gamma=gamma
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 4. Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )
    
    # 5. Early stopping mechanism
    early_stopping = EarlyStopping(patience=50, verbose=True)
    
    print(f"\n[Training] Starting training...")
    print(f"  Max Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  HAZ Multiplier: {haz_multiplier}")
    print(f"  Loss Weights (α, β, γ): ({alpha}, {beta}, {gamma})")
    print(f"  LR Scheduler: ReduceLROnPlateau (patience=20, factor=0.5)")
    print(f"  Early Stopping: patience=50\n")
    
    # 6. Training loop
    history = {'train_loss': [], 'valid_loss': [], 'lr': []}
    best_model_state = None
    best_valid_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        train_epoch_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        
        avg_train_loss = train_epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # --- Validation phase ---
        model.eval()
        valid_epoch_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                v_loss = criterion(outputs, targets)
                valid_epoch_loss += v_loss.item()
            
        avg_valid_loss = valid_epoch_loss / len(valid_loader)
        history['valid_loss'].append(avg_valid_loss)
        
        # Record current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # --- Learning rate scheduling ---
        scheduler.step(avg_valid_loss)
        
        # --- Save best model ---
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
        
        # --- Early stopping check ---
        early_stopping(avg_valid_loss, epoch + 1)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Valid: {avg_valid_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
        
        if early_stopping.early_stop:
            print(f"\nTraining stopped early at Epoch {epoch+1}")
            break
    
    # 7. Restore and save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[Best Model] Epoch {best_epoch}, Valid Loss: {best_valid_loss:.6f}")
    
    save_path = os.path.join(save_dir, 'baseline_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"[Save] Best model saved to: {save_path}")
    
    # 8. Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss curve
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['valid_loss'], label='Validation Loss', linewidth=2)
    ax1.axvline(best_epoch - 1, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    
    # Learning rate curve
    ax2.plot(history['lr'], label='Learning Rate', linewidth=2, color='green')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.suptitle('Baseline Training with Early Stopping & LR Scheduler', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    curve_path = os.path.join(save_dir, 'baseline_loss_curve.png')
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    print(f"[Save] Training curve saved to: {curve_path}")
    plt.close()
    
    print("\n✓ Baseline training completed!")
    return history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TripleHead Baseline Training')
    parser.add_argument('--epochs', type=int, default=1000, help='Maximum training epochs (with early stopping)')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--haz', type=int, default=10, help='HAZ weight multiplier')
    parser.add_argument('--channels', type=int, default=128, choices=[32, 64, 128],
                        help='Model base channels')
    
    args = parser.parse_args()
    
    train_baseline(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        haz_multiplier=args.haz,
        base_channels=args.channels
    )
