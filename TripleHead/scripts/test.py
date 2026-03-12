# -*- coding: utf-8 -*-
"""
Testing and Prediction Script
Use the trained model to predict the plastic strain distribution for any thickness
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import WeldingDataset
from models.triple_head_model import TripleHeadPEModel


# ========== User Configuration Area ==========
# Model directory: contains best_model.pth and best_params.json
MODEL_DIR = r'D:\CNN_InSM_New\TripleHead\results\Web15mm'

CONFIG = {
    'thickness': 17,  # Default prediction thickness (mm)
    'model_path': os.path.join(MODEL_DIR, 'best_model.pth'),
    'params_path': os.path.join(MODEL_DIR, 'best_params.json'),
    'data_dir': r'D:\CNN_InSM_New\data_input\numpy_data\from_interpo\Web15mm',
    'save_dir': r'D:\CNN_InSM_New\TripleHead\results\Web15mm'
}

# Dynamically load hyperparameters
import json
with open(CONFIG['params_path'], 'r') as f:
    BEST_PARAMS = json.load(f)
    CONFIG['base_channels'] = BEST_PARAMS.get('base_channels', 64)
    print(f"[Config] Loaded base_channels = {CONFIG['base_channels']} from best_params.json")
# =====================================================



def predict_thickness(thickness_mm=CONFIG['thickness'],
                      model_path=CONFIG['model_path'],
                      data_dir=CONFIG['data_dir'],
                      base_channels=CONFIG['base_channels'],
                      save_dir=CONFIG['save_dir']):
    """
    Predict plastic strain distribution for a specified thickness
    
    Args:
        thickness_mm: Thickness (mm)
        model_path: Model weights path
        data_dir: Data directory (used to get normalization parameters)
        base_channels: Model channels
        save_dir: Result save path (if None, do not save)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load dataset to get normalization parameters
    print(f"[Load] Loading normalization parameters...")
    dataset = WeldingDataset(data_dir, augment=False)
    global_max_t = dataset.global_max_t
    global_max_pe = dataset.global_max_pe
    print(f"  Max Thickness: {global_max_t}mm")
    print(f"  Max PE: {global_max_pe}")
    
    # 2. Load model
    print(f"\n[Model] Loading model: {model_path}")
    model = TripleHeadPEModel(base_channels=base_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  Total parameters: {model.get_num_parameters():,}")
    
    # 3. Prepare inputs
    normalized_thickness = torch.tensor([[thickness_mm / global_max_t]], dtype=torch.float32).to(device)
    
    # 4. Predict
    print(f"\n[Predict] Predicting thickness T = {thickness_mm}mm...")
    with torch.no_grad():
        output = model(normalized_thickness)  # [1, 3, 256, 256]
    
    # Denormalize
    # global_max_pe currently has shape (3,), needs to be reshaped to broadcast correctly with (3, 256, 256)
    output_np = output.cpu().numpy()[0] * global_max_pe.reshape(3, 1, 1)  # [3, 256, 256]
    pe11, pe22, pe33 = output_np[0], output_np[1], output_np[2]
    
    print(f"  PE11 Range: [{pe11.min():.6f}, {pe11.max():.6f}]")
    print(f"  PE22 Range: [{pe22.min():.6f}, {pe22.max():.6f}]")
    print(f"  PE33 Range: [{pe33.min():.6f}, {pe33.max():.6f}]")
    
    # 5. Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    extent = [-0.02, 0.02, -0.02, 0.02]  # Physical coordinate range
    
    components = [
        ('PE11 (Longitudinal)', pe11),
        ('PE22 (Transverse)', pe22),
        ('PE33 (Thickness)', pe33)
    ]
    
    for ax, (title, data) in zip(axes, components):
        im = ax.imshow(data, extent=extent, origin='lower', cmap='jet')
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title(f'{title}\nT = {thickness_mm}mm', fontsize=12, fontweight='bold')
        ax.axhline(0, color='white', linestyle=':', linewidth=1, alpha=0.7)
        ax.axvline(0, color='white', linestyle=':', linewidth=1, alpha=0.7)
        plt.colorbar(im, ax=ax, label='Plastic Strain')
    
    plt.suptitle(f'TripleHead Model Prediction - Thickness {thickness_mm}mm',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'prediction_T_{thickness_mm}mm.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[Save] Prediction results saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return output_np


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TripleHead Model Prediction')
    parser.add_argument('--thickness', type=float, default=CONFIG['thickness'],
                        help=f'Thickness (mm), default: {CONFIG["thickness"]}')
    parser.add_argument('--model', type=str, default=CONFIG['model_path'],
                        help=f'Model path, default: {CONFIG["model_path"]}')
    parser.add_argument('--channels', type=int, default=CONFIG['base_channels'],
                        choices=[32, 64, 128], help=f'Model channels, default: {CONFIG["base_channels"]}')
    parser.add_argument('--save_dir', type=str, default=CONFIG['save_dir'],
                        help=f'Result save path, default: {CONFIG["save_dir"]}')
    
    args = parser.parse_args()
    
    predict_thickness(
        thickness_mm=args.thickness,
        model_path=args.model,
        base_channels=args.channels,
        save_dir=args.save_dir
    )
