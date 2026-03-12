# -*- coding: utf-8 -*-
"""
WeldingDataset for TripleHead Model
Welding dataset that supports unified normalization parameters
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os


class WeldingDataset(Dataset):
    """
    Welding plastic strain dataset
    
    Features:
    1. Supports manually specifying normalization parameters (to ensure train/val consistency)
    2. Supports data augmentation (horizontal flip)
    3. Automatically parses thickness information from filenames (supports integers and decimal points formats)
    
    Args:
        data_dir: Path to the data directory
        augment: Whether to enable data augmentation
        file_list: List of filenames (scans the whole directory if None)
        manual_max_pe: Manually specified max PE value (for normalization)
        manual_max_t: Manually specified max thickness value (for normalization)
    """
    
    def __init__(self, data_dir, augment=True, file_list=None,
                 manual_max_pe=None, manual_max_t=None):
        self.data_dir = data_dir
        self.augment = augment
        self.file_list = file_list if file_list is not None else \
                         [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        # --- Core modifications: enable independent normalization for components ---
        if manual_max_pe is not None and manual_max_t is not None:
            # Ensure manual_max_pe is an array or list of length 3
            self.global_max_pe = np.array(manual_max_pe)
            self.global_max_t = manual_max_t
            print(f"[Dataset] Using specified normalization parameters: Max PE = {self.global_max_pe}, Max T = {self.global_max_t}mm")
        else:
            self.global_max_pe = np.zeros(3) # Records the max values of PE11, PE22, PE33
            self.global_max_t = 0.0
            print("[Dataset] Scanning dataset to obtain component-independent dynamic normalization parameters...")
            for f in self.file_list:
                data = np.load(os.path.join(data_dir, f)) # [H, W, 3]
                
                # Fetch absolute max values for each of the 3 components
                for i in range(3):
                    current_max = np.max(np.abs(data[:,:,i]))
                    if current_max > self.global_max_pe[i]:
                        self.global_max_pe[i] = current_max
                
                # Supports 'Plate19p0mm.npy' format
                # Removes 'Plate', '.npy', and 'mm', replaces 'p' with '.'
                t_str = f.replace('Plate', '').replace('.npy', '').replace('mm', '')
                t_str = t_str.replace('p', '.')
                try:
                    t_val = float(t_str)
                    if t_val > self.global_max_t:
                        self.global_max_t = t_val
                except ValueError:
                    pass
            
            # Add 5% margin
            self.global_max_pe *= 1.05
            print(f"[Dataset] Self-scan completed: Max PE = {self.global_max_pe}, Max T = {self.global_max_t}mm")

    def __len__(self):
        return len(self.file_list) * 2 if self.augment else len(self.file_list)

    def __getitem__(self, idx):
        is_flip = False
        if self.augment and idx >= len(self.file_list):
            file_idx = idx - len(self.file_list)
            is_flip = True
        else:
            file_idx = idx
            
        file_name = self.file_list[file_idx]
        image = np.load(os.path.join(self.data_dir, file_name)).astype(np.float32) # [H, W, 3]
        
        # Dynamically extract thickness and normalize with global max thickness
        thickness_str = file_name.replace('Plate', '').replace('.npy', '').replace('mm', '')
        thickness_str = thickness_str.replace('p', '.')
        thickness = float(thickness_str)
        input_feat = torch.tensor([thickness / self.global_max_t], dtype=torch.float32)
        
        if is_flip:
            image = np.flip(image, axis=1).copy()
        
        # Independently normalize using the respective Max PE for each of the three components
        # The size of 'image' is [H, W, 3], perform division on the last axis
        image = image / self.global_max_pe.astype(np.float32)
        target_tensor = torch.from_numpy(image).permute(2, 0, 1)  # [3, 256, 256]
        
        return input_feat, target_tensor


def get_train_val_datasets(data_dir, valid_ratio=0.2, random_seed=42, holdout_samples=None):
    """
    Create training and validation sets, ensuring uniform normalization parameters
    
    Args:
        data_dir: Data directory path
        valid_ratio: Ratio to use for validation set
        random_seed: Random seed
        holdout_samples: List of sample identifiers to exclude (e.g., ['24.0mm', '24.5mm'])
    
    Returns:
        train_dataset, valid_dataset: Training and validation sets
    """
    import random
    
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    # Exclude holdout samples
    if holdout_samples:
        original_count = len(all_files)
        all_files = [f for f in all_files if not any(h in f for h in holdout_samples)]
        excluded_count = original_count - len(all_files)
        print(f"\\n[Holdout] Excluded {excluded_count} samples, identifiers: {holdout_samples}")
    
    all_files.sort()
    random.seed(random_seed)
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * (1 - valid_ratio))
    train_files = all_files[:split_idx]
    valid_files = all_files[split_idx:]

    
    # Extract thickness distribution for review
    def extract_thicknesses(files):
        t_vals = []
        for f in files:
            # Supports 'Plate19p0mm.npy' format
            t_str = f.replace('Plate', '').replace('.npy', '').replace('mm', '')
            t_str = t_str.replace('p', '.')
            try:
                t_vals.append(float(t_str))
            except ValueError:
                pass
        return sorted(list(set(t_vals)))

    train_ts = extract_thicknesses(train_files)
    valid_ts = extract_thicknesses(valid_files)

    print(f"\n[Split] Compile Train Set: {len(train_files)} samples")
    print(f"  Includes thicknesses (mm): {train_ts}")
    print(f"[Split] Compile Validation Set: {len(valid_files)} samples")
    print(f"  Includes thicknesses (mm): {valid_ts}\n")
    
    # Create the training set first so that it triggers 'self-scan'
    train_dataset = WeldingDataset(data_dir, augment=True, file_list=train_files)
    
    # Extract baseline parameters from the training set
    t_max_base = train_dataset.global_max_t
    pe_max_base = train_dataset.global_max_pe
    
    # Force the validation set to use the exact same parameters
    valid_dataset = WeldingDataset(data_dir, augment=False, file_list=valid_files,
                                   manual_max_pe=pe_max_base,
                                   manual_max_t=t_max_base)
    
    return train_dataset, valid_dataset


if __name__ == "__main__":
    # Test dataset
    data_dir = r'F:\InSM_python\CNN-InSM\CNN_Dataset\ChangeWeb\Web10mm\sigma0.0'
    if os.path.exists(data_dir):
        print("=== Test WeldingDataset ===")
        train_ds, valid_ds = get_train_val_datasets(data_dir)
        
        print(f"\nTraining set size: {len(train_ds)}")
        print(f"Validation set size: {len(valid_ds)}")
        
        x, y = train_ds[0]
        print(f"\nSample shape:")
        print(f"  Input (thickness): {x.shape}")
        print(f"  Target (PE components): {y.shape}")
        print(f"  Thickness value: {x.item() * train_ds.global_max_t:.1f}mm")
        print("\n✓ Dataset test passed")
