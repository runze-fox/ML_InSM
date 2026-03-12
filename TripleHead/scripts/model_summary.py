# -*- coding: utf-8 -*-
"""
Model Parameter Summary Script
Calculate and display details of the parameter distribution in the TripleHead model (for papers/reports)
"""
import os
import sys
import json
import torch

# Add path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.triple_head_model import TripleHeadPEModel

# Path configuration
PARAMS_PATH = r'F:\InSM_python\CNN-InSM\TripleHead\models\model_for_test\best_params.json'

def count_parameters(model):
    """Count the number of parameters in each part of the model"""
    summary = []
    
    # 1. Evaluate the shared backbone
    backbone_params = sum(p.numel() for p in model.fc.parameters())
    summary.append(["Shared Backbone (FC Layers)", backbone_params])
    
    # 2. Evaluate the three independent heads
    heads = [
        ("Head PE11 (Longitudinal)", model.head_pe11),
        ("Head PE22 (Transverse)", model.head_pe22),
        ("Head PE33 (Thickness)", model.head_pe33)
    ]
    
    for name, head in heads:
        params = sum(p.numel() for p in head.parameters())
        summary.append([name, params])
    
    return summary

def main():
    # Load optimal channels configuration
    if not os.path.exists(PARAMS_PATH):
        print(f"[Error] Parameter file not found: {PARAMS_PATH}")
        base_ch = 32  # Default fallback
    else:
        with open(PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
            base_ch = best_params.get('base_channels', 32)
    
    # Instantiate the model
    model = TripleHeadPEModel(base_channels=base_ch)
    total_params = model.get_num_parameters()
    summary_data = count_parameters(model)
    
    # Print in Markdown table format (for easy pasting into papers/README)
    print(f"\n### TripleHead Parameter Summary (base_channels={base_ch})")
    print("| Module Name | Parameters Quantity | Ratio (%) |")
    print("| :--- | :---: | :---: |")
    
    for name, count in summary_data:
        percentage = (count / total_params) * 100
        print(f"| {name} | {count:,} | {percentage:.2f}% |")
    
    print(f"| **Total** | **{total_params:,}** | **100.00%** |")
    print(f"\n*Note: This model uses an independent PE33 decoding branch for thickness direction prediction, total parameter count is approximately {total_params/1e6:.2f} M.*")

if __name__ == "__main__":
    main()