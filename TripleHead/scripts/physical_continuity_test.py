# -*- coding: utf-8 -*-
"""
Physical Continuity Test Script
Verify the physical continuity and prediction accuracy of the model under different thicknesses
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

# Add path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import WeldingDataset
from models.triple_head_model import TripleHeadPEModel


# ========== User Configuration Area ==========
# Model directory: contains best_model.pth and best_params.json
MODEL_DIR = r'D:\CNN_InSM_New\TripleHead\results\Web9mm\20260310_200802_tuning'

CONFIG = {
    # Test thickness list: integers are train set points, numbers ending in 0.5mm are test set points
    'test_thicknesses': [8.5, 12.5, 17.5, 22.5],
    #'test_thicknesses': [12,17,22],
    
    # Path configuration
    'model_path': os.path.join(MODEL_DIR, 'best_model.pth'),
    'params_path': os.path.join(MODEL_DIR, 'best_params.json'),
    'data_dir': r'D:\CNN_InSM_New\data_input\numpy_data\from_interpo\Web9mm',
    'save_dir': r'D:\CNN_InSM_New\TripleHead\results\Web9mm\continuity_test'
}

# Dynamically load hyperparameters
with open(CONFIG['params_path'], 'r') as f:
    BEST_PARAMS = json.load(f)
    CONFIG['base_channels'] = BEST_PARAMS.get('base_channels', 64)
    print(f"[Config] Loaded base_channels = {CONFIG['base_channels']} from best_params.json")
# =====================================================


def load_ground_truth(data_dir, thickness_mm, global_max_pe):
    """
    Load the corresponding ground truth .npy file given the thickness and denormalize it
    
    Args:
        data_dir: Data directory
        thickness_mm: Thickness (mm)
        global_max_pe: Global max PE value (used for denormalization)
    
    Returns:
        ground_truth: Ground truth array of shape [3, 256, 256] (raw physical values)
    """
    # Match filename based on thickness (supports 19.0 or 19p0 formats)
    thickness_str = f"{thickness_mm}"
    
    # Search for a matching file
    for f in os.listdir(data_dir):
        if not f.endswith('.npy'):
            continue
        # Extract thickness from filename
        try:
            if '_T_' in f:
                t_str = f.split('_T_')[1].split('mm')[0].replace('p', '.')
            else:
                t_str = f.replace('Plate', '').replace('.npy', '').replace('mm', '').replace('p', '.')
                
            if float(t_str) == thickness_mm:
                file_path = os.path.join(data_dir, f)
                data = np.load(file_path).astype(np.float32)
                # data shape: [256, 256, 3] -> convert to [3, 256, 256]
                data = np.transpose(data, (2, 0, 1))
                print(f"  [GT] Loaded: {f}")
                return data  # Note: dataset is already normalized, returning raw values here
        except (IndexError, ValueError):
            continue
    
    raise FileNotFoundError(f"Could not find ground truth file for thickness {thickness_mm}mm")


def predict_single(model, thickness_mm, global_max_t, device):
    """
    Predict PE distribution for a single thickness
    
    Returns:
        output: Predicted array of shape [3, 256, 256] (normalized values)
    """
    normalized_t = torch.tensor([[thickness_mm / global_max_t]], dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(normalized_t)  # [1, 3, 256, 256]
    return output.cpu().numpy()[0]  # [3, 256, 256]


def calculate_rmse(pred, gt, mask=None):
    """
    Calculate RMSE
    
    Args:
        pred, gt: Predicted and ground truth values
        mask: Optional mask for valid region
    """
    diff = pred - gt
    if mask is not None:
        diff = diff[mask > 0]
    return np.sqrt(np.mean(diff ** 2))


def calculate_relative_error(pred, gt, epsilon=1e-8):
    """
    Calculate relative error percentage
    """
    gt_abs = np.abs(gt)
    gt_abs[gt_abs < epsilon] = epsilon  # Avoid division by zero
    rel_err = np.abs(pred - gt) / gt_abs
    return np.mean(rel_err) * 100  # Percentage


def visualize_comparison(gt, pred, thickness_mm, save_path):
    """
    Visualization A: Generate 3x3 comparison plots (GT/Pred/Diff)
    
    Args:
        gt: [3, 256, 256] Ground truth
        pred: [3, 256, 256] Prediction
        thickness_mm: Thickness
        save_path: Save path
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    extent = [-0.02, 0.02, -0.02, 0.02]
    component_names = ['PE11', 'PE22', 'PE33']
    row_titles = ['Ground Truth (Abaqus)', 'Model Prediction', 'Difference (Pred - GT)']
    
    # Calculate difference
    diff = pred - gt
    
    for col, (comp_name, gt_ch, pred_ch, diff_ch) in enumerate(zip(
            component_names, gt, pred, diff)):
        
        # Unified color scaling
        vmin_data = min(gt_ch.min(), pred_ch.min())
        vmax_data = max(gt_ch.max(), pred_ch.max())
        vmax_diff = max(abs(diff_ch.min()), abs(diff_ch.max()))
        
        # Row 1: Ground truth
        im0 = axes[0, col].imshow(gt_ch, extent=extent, origin='lower', 
                                   cmap='jet', vmin=vmin_data, vmax=vmax_data)
        axes[0, col].set_title(f'{comp_name} - GT', fontsize=11, fontweight='bold')
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046)
        
        # Row 2: Prediction
        im1 = axes[1, col].imshow(pred_ch, extent=extent, origin='lower', 
                                   cmap='jet', vmin=vmin_data, vmax=vmax_data)
        axes[1, col].set_title(f'{comp_name} - Pred', fontsize=11, fontweight='bold')
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046)
        
        # Row 3: Difference (using seismic map, centered at 0)
        im2 = axes[2, col].imshow(diff_ch, extent=extent, origin='lower', 
                                   cmap='seismic', vmin=-vmax_diff, vmax=vmax_diff)
        rmse = calculate_rmse(pred_ch, gt_ch)
        axes[2, col].set_title(f'{comp_name} - Diff (RMSE: {rmse:.2e})', 
                               fontsize=11, fontweight='bold')
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046)
    
    # Add row labels
    for ax, title in zip(axes[:, 0], row_titles):
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
    
    # Add axis labels
    for ax in axes.flat:
        ax.set_xlabel('X (m)', fontsize=9)
        ax.tick_params(labelsize=8)
    
    plt.suptitle(f'Physical Continuity Test - Thickness {thickness_mm}mm', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Save] Comparison plot saved: {save_path}")


def visualize_error_trend(results, save_path):
    """
    Visualization B: Error-Thickness trend line chart
    
    Args:
        results: List of dictionaries containing errors for each thickness
        save_path: Save path
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    component_names = ['PE11', 'PE22', 'PE33']
    
    # Separate train set points (integers) and test set points (0.5mm)
    train_results = [r for r in results if r['thickness'] == int(r['thickness'])]
    test_results = [r for r in results if r['thickness'] != int(r['thickness'])]
    
    for idx, comp in enumerate(component_names):
        ax = axes[idx]
        
        # Plot training set points (circles, blue)
        if train_results:
            train_t = [r['thickness'] for r in train_results]
            train_rmse = [r[f'rmse_{comp.lower()}'] for r in train_results]
            ax.plot(train_t, train_rmse, 'o-', color='blue', markersize=8, 
                    linewidth=2, label='Train Set (Integer mm)')
        
        # Plot test set points (triangles, red)
        if test_results:
            test_t = [r['thickness'] for r in test_results]
            test_rmse = [r[f'rmse_{comp.lower()}'] for r in test_results]
            ax.plot(test_t, test_rmse, '^--', color='red', markersize=8, 
                    linewidth=2, label='Test Set (0.5 mm)')
        
        ax.set_xlabel('Thickness (mm)', fontsize=11)
        ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title(f'{comp} Error vs Thickness', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Model Prediction Error Trend', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Save] Error trend plot saved: {save_path}")


def run_continuity_test():
    """
    Main test function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create results directory
    if not os.path.exists(CONFIG['save_dir']):
        os.makedirs(CONFIG['save_dir'])
    
    # 1. Load dataset to get normalization parameters
    print("\n" + "=" * 60)
    print("Physical Continuity Test")
    print("=" * 60)
    print(f"\n[Load] Loading normalization parameters...")
    dataset = WeldingDataset(CONFIG['data_dir'], augment=False)
    global_max_t = dataset.global_max_t
    global_max_pe = dataset.global_max_pe
    print(f"  Max Thickness: {global_max_t}mm")
    print(f"  Max PE: {global_max_pe}")
    
    # 2. Load model
    print(f"\n[Model] Loading model: {CONFIG['model_path']}")
    model = TripleHeadPEModel(base_channels=CONFIG['base_channels']).to(device)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device, weights_only=True))
    model.eval()
    print(f"  Total parameters: {model.get_num_parameters():,}")
    
    # 3. Iterate through test thicknesses
    results = []
    print(f"\n[Test] Test thickness list: {CONFIG['test_thicknesses']}")
    
    for thickness in CONFIG['test_thicknesses']:
        print(f"\n--- Processing thickness {thickness}mm ---")
        
        # Load ground truth
        try:
            gt = load_ground_truth(CONFIG['data_dir'], thickness, global_max_pe)
        except FileNotFoundError as e:
            print(f"  [Warning] {e}")
            continue
        
        # Predict (model output is normalized, requires denormalization)
        pred = predict_single(model, thickness, global_max_t, device)
        pred = pred * global_max_pe.reshape(3, 1, 1)  # Denormalize
        
        # Calculate RMSE for each component
        rmse_pe11 = calculate_rmse(pred[0], gt[0])
        rmse_pe22 = calculate_rmse(pred[1], gt[1])
        rmse_pe33 = calculate_rmse(pred[2], gt[2])
        
        print(f"  RMSE - PE11: {rmse_pe11:.6f}, PE22: {rmse_pe22:.6f}, PE33: {rmse_pe33:.6f}")
        
        results.append({
            'thickness': thickness,
            'rmse_pe11': rmse_pe11,
            'rmse_pe22': rmse_pe22,
            'rmse_pe33': rmse_pe33
        })
        
        # Generate Visualization A: Comparison plot
        save_path = os.path.join(CONFIG['save_dir'], f'comparison_T_{thickness}mm.png')
        visualize_comparison(gt, pred, thickness, save_path)
        
        # Save predicted tensor as .npy format for strain_to_params.py parsing
        npy_dir = CONFIG['save_dir']
        os.makedirs(npy_dir, exist_ok=True)
        npy_path = os.path.join(npy_dir, f'predicted_strain_T{thickness}.npy')
        np.save(npy_path, pred)
        print(f"  [Save] Predicted tensor saved to: {npy_path}")
    
    # 4. Generate Visualization B: Error trend plot
    if results:
        trend_path = os.path.join(CONFIG['save_dir'], 'error_trend.png')
        visualize_error_trend(results, trend_path)
        
        # Save numerical results
        results_path = os.path.join(CONFIG['save_dir'], 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[Save] Test results saved: {results_path}")
    
    print("\n" + "=" * 60)
    print("Physical continuity test completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_continuity_test()
