# -*- coding: utf-8 -*-
"""
Optuna Hyperparameter Tuning Script
Use Optuna to automatically search for optimal hyperparameters
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner
import os
import sys
import json
from datetime import datetime

# Low-level computation optimization: allow CUDA to find the best convolution algorithm for current hardware
torch.backends.cudnn.benchmark = True

# Add path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_train_val_datasets
from models.triple_head_model import TripleHeadPEModel
from models.losses import ComponentWeightedHAZLoss


# Global configuration
DATA_DIR = r'D:\CNN_InSM_New\data_input\numpy_data\from_interpo\Web9mm'
SAVE_DIR = r'D:\CNN_InSM_New\TripleHead\results\Web9mm'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training depth control parameters
N_TRIALS = 5       # Total number of trials for Optuna "audition" phase
TRIAL_EPOCHS = 150   # Number of epochs trained per trial in the "audition phase", for quick evaluation
FINAL_EPOCHS = 1000  # Number of complete epochs for the "final training" after selecting the best parameters

# Hardware performance parameters
N_JOBS = 1          # Number of parallel trials (optimized for 8GB VRAM)
BATCH_SIZE = 8      # DataLoader batch size
NUM_WORKERS = 0     # Number of DataLoader worker threads (set to 0 on Windows to avoid multi-processing conflicts)

# Holdout set (these samples are excluded from train/val split and used for subsequent independent testing)
# Example: ['24.0mm', '24.5mm'] will exclude any samples with filenames containing these strings
HOLDOUT_SAMPLES = ['8.5mm','12.5mm','17.5mm','22.5mm']  # Default is empty, no samples excluded
# Global dataset variables (initialized in main)
TRAIN_DATASET = None
VALID_DATASET = None


def check_gpu_status():
    """
    Check and print GPU status information
    """
    print("\n" + "=" * 60)
    print("GPU Status Check")
    print("=" * 60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved_mem = torch.cuda.memory_reserved(0) / (1024**3)
        allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
        free_mem = total_mem - reserved_mem
        
        print(f"  GPU Model: {gpu_name}")
        print(f"  Total VRAM: {total_mem:.2f} GB")
        print(f"  Reserved: {reserved_mem:.2f} GB")
        print(f"  Allocated: {allocated_mem:.2f} GB")
        print(f"  Free VRAM: {free_mem:.2f} GB")
        print(f"  cuDNN Benchmark: {'Enabled' if torch.backends.cudnn.benchmark else 'Disabled'}")
    else:
        print("  [Warning] CUDA unavailable, falling back to CPU for training")
    
    print("=" * 60 + "\n")


def load_datasets():
    """
    Load train and validation datasets (called only once in the main process)
    """
    global TRAIN_DATASET, VALID_DATASET
    # get_train_val_datasets internally contains detailed print info about the split
    TRAIN_DATASET, VALID_DATASET = get_train_val_datasets(
        DATA_DIR, valid_ratio=0.2, holdout_samples=HOLDOUT_SAMPLES
    )


def objective(trial):
    """
    Optuna optimization objective function
    
    Args:
        trial: Optuna trial object
    
    Returns:
        float: Validation set loss (lower is better)
    """
    # ========== 1. Parameter Sampling ==========
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    wd = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    haz_mult = trial.suggest_int('haz_multiplier', 1, 10)
    # haz_mult = trial.suggest_int('haz_multiplier', 5, 50)
    base_ch = trial.suggest_categorical('base_channels', [64, 128, 256])
    
    # Component loss weights
    alpha = trial.suggest_uniform('alpha_pe11', 0.5, 2.0)
    beta = trial.suggest_uniform('beta_pe22', 0.5, 2.0)
    gamma = trial.suggest_uniform('gamma_pe33', 0.5, 2.0)
    
    print(f"\n[Trial {trial.number}] Parameter configuration:")
    print(f"  LR: {lr:.2e}, WD: {wd:.2e}, HAZ: {haz_mult}")
    print(f"  Channels: {base_ch}, Loss Weights: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")
    
    # ========== VRAM Monitoring ==========
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"  [VRAM] Reserved at start of Trial: {reserved:.2f} GB")
    
    # ========== 2. Build model and optimizer ==========
    model = TripleHeadPEModel(base_channels=base_ch).to(DEVICE)
    criterion = ComponentWeightedHAZLoss(haz_mult, alpha, beta, gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    # ========== 3. Data loader (optimized) ==========
    train_loader = DataLoader(
        TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    valid_loader = DataLoader(
        VALID_DATASET, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # ========== 4. Training loop (with pruning) ==========
    for epoch in range(TRIAL_EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                v_loss = criterion(outputs, targets)
                valid_loss += v_loss.item()
        
        avg_valid_loss = valid_loss / len(valid_loader)
        
        # --- Report intermediate results and check pruning ---
        trial.report(avg_valid_loss, epoch)
        
        # Early stopping if the current trial performs poorly
        if trial.should_prune():
            print(f"  [Pruned] Trial {trial.number} pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()
        
        if (epoch + 1) % 10 == 0:
            print(f"  [T{trial.number}] Epoch [{epoch+1}/{TRIAL_EPOCHS}] Train: {avg_train_loss:.6f}, Valid: {avg_valid_loss:.6f}")
    
    # ========== 5. Clear VRAM ==========
    if torch.cuda.is_available():
        reserved_before = torch.cuda.memory_reserved(0) / (1024**3)
    
    del model, optimizer
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        reserved_after = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"  [VRAM] After clearing: {reserved_before:.2f} GB -> {reserved_after:.2f} GB")
    
    print(f"[Trial {trial.number}] Final validation loss: {avg_valid_loss:.6f}")
    return avg_valid_loss


def run_optimization(n_trials=50, study_name='triplehead_optuna'):
    """
    Run Optuna optimization
    
    Args:
        n_trials: Number of trials to attempt
        study_name: Name of the study
    """
    # Generate result directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(SAVE_DIR, f'{timestamp}_tuning')
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Output Directory] {run_dir}")
    
    # Create Optuna Study
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name=study_name
    )
    
    print(f"\n{'='*60}")
    print(f"Starting Optuna hyperparameter optimization")
    print(f"  Device: {DEVICE}")
    print(f"  Number of Trials: {n_trials}")
    print(f"  Epochs per Trial: {TRIAL_EPOCHS}")
    print(f"  Final Training Epochs: {FINAL_EPOCHS}")
    print(f"  Parallel Jobs: {N_JOBS}")
    print(f"  Batch Size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    print(f"{'='*60}\n")
    
    # Execute optimization (multi-task parallel processing)
    study.optimize(objective, n_trials=n_trials, n_jobs=N_JOBS)
    
    # ========== Output Results ==========
    print(f"\n{'='*60}")
    print("Optimization completed!")
    print(f"{'='*60}")
    
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save the best parameters to JSON
    best_params_path = os.path.join(run_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nBest parameters saved to: {best_params_path}")
    
    # ========== Retrain full model with optimal parameters ==========
    print(f"\n{'='*60}")
    print("Retraining the complete model with the best parameters...")
    print(f"{'='*60}\n")
    
    best_params = study.best_params
    final_model = TripleHeadPEModel(base_channels=best_params['base_channels']).to(DEVICE)
    final_criterion = ComponentWeightedHAZLoss(
        haz_multiplier=best_params['haz_multiplier'],
        alpha=best_params['alpha_pe11'],
        beta=best_params['beta_pe22'],
        gamma=best_params['gamma_pe33']
    )
    final_optimizer = optim.Adam(
        final_model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        final_optimizer, mode='min', factor=0.5, patience=20
    )
    
    train_loader = DataLoader(
        TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    valid_loader = DataLoader(
        VALID_DATASET, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Early stopping and saving the best model
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    early_stop_patience = 50
    model_path = os.path.join(run_dir, 'best_model.pth')
    
    print(f"[Final Configuration] Max Epochs: {FINAL_EPOCHS}, Early Stop Patience: {early_stop_patience}")
    print(f"[Final Configuration] LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=20)\n")
    
    # Training Loop
    for epoch in range(FINAL_EPOCHS):
        # --- Training ---
        final_model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = final_model(inputs)
            loss = final_criterion(outputs, targets)
            
            final_optimizer.zero_grad()
            loss.backward()
            final_optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        final_model.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = final_model(inputs)
                valid_loss += final_criterion(outputs, targets).item()
        
        avg_val_loss = valid_loss / len(valid_loader)
        
        # --- Learning Rate Scheduling ---
        scheduler.step(avg_val_loss)
        current_lr = final_optimizer.param_groups[0]['lr']
        
        # --- Check for new best record ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            # Save the best model immediately
            torch.save(final_model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
        
        # --- Output Logs (Every 20 Epochs) ---
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{FINAL_EPOCHS}] "
                  f"Train: {avg_train_loss:.6f}, Valid: {avg_val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} (Ep {best_epoch}) | "
                  f"LR: {current_lr:.2e} | NoImprove: {epochs_no_improve}")
        
        # --- Early Stopping Check ---
        if epochs_no_improve >= early_stop_patience:
            print(f"\n[Early Stop] No improvement for {early_stop_patience} consecutive epochs, training terminated early")
            print(f"[Early Stop] Best validation loss: {best_val_loss:.6f} (Epoch {best_epoch})")
            break
    
    print(f"\n{'='*60}")
    print(f"Final training completed!")
    print(f"  Best validation loss: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"  Model saved to: {model_path}")
    print(f"{'='*60}")
    
    return study



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TripleHead Optuna Tuning')
    parser.add_argument('--n_trials', type=int, default=N_TRIALS, help='Number of Optuna trials')
    parser.add_argument('--trial_epochs', type=int, default=TRIAL_EPOCHS, help='Training epochs per trial')
    parser.add_argument('--final_epochs', type=int, default=FINAL_EPOCHS, help='Epochs for final training')
    parser.add_argument('--n_jobs', type=int, default=N_JOBS, help='Number of parallel trials')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='DataLoader batch size')
    
    args = parser.parse_args()
    
    # Synchronize global configurations
    N_TRIALS = args.n_trials
    TRIAL_EPOCHS = args.trial_epochs
    FINAL_EPOCHS = args.final_epochs
    N_JOBS = args.n_jobs
    BATCH_SIZE = args.batch_size
    
    # Check GPU status
    check_gpu_status()
    
    # Load dataset (only executed once in main process)
    load_datasets()
    
    study = run_optimization(n_trials=N_TRIALS)
