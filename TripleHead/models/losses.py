# -*- coding: utf-8 -*-
"""
Loss Functions for TripleHead Model
Component-weighted + HAZ region-weighted loss function
"""
import torch
import torch.nn as nn


class ComponentWeightedHAZLoss(nn.Module):
    """
    Component-weighted HAZ loss function
    
    Features:
    1. HAZ regional weighting: The Heat-Affected Zone (near the weld) has higher error weights
    2. Component independent weights: PE11, PE22, PE33 can have different loss weights
    
    Design Concept:
    - PE11 (Longitudinal strain) is usually the largest and the main optimization target
    - PE22 (Transverse strain) is secondary
    - PE33 (Thickness direction) is the smallest but noisy, might need higher or lower weights
    
    Args:
        haz_multiplier: Weight multiplier for the HAZ region (default 10.0)
        alpha: Loss weight for PE11 component (default 1.0)
        beta: Loss weight for PE22 component (default 1.0)
        gamma: Loss weight for PE33 component (default 1.0)
    """
    
    def __init__(self, haz_multiplier=10.0, alpha=1.0, beta=1.0, gamma=1.0):
        super(ComponentWeightedHAZLoss, self).__init__()
        self.haz_multiplier = haz_multiplier
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Create spatial weight mask (256, 256)
        # Based on ROI (-0.02, 0.02) and origin at the top surface of the base plate, the central area is the HAZ
        self.weight_mask = torch.ones(256, 256)
        
        # Define the HAZ region (pixel indices): near the center weld
        # These indices correspond to the weld and heat-affected zone in physical space
        self.weight_mask[96:160, 64:192] = haz_multiplier
        
        # Non-trainable parameter (will be automatically transferred when moving to GPU)
        self.register_buffer('_weight_mask_buffer', self.weight_mask)
    
    def forward(self, pred, target):
        """
        Calculate weighted loss
        
        Args:
            pred: Model predictions [batch_size, 3, 256, 256]
            target: Ground truth [batch_size, 3, 256, 256]
        
        Returns:
            loss: Scalar loss value
        """
        # Ensure mask is on the correct device
        mask = self._weight_mask_buffer.to(pred.device)
        
        # Calculate weighted MSE for the three components respectively
        loss_pe11 = torch.mean(mask * (pred[:, 0] - target[:, 0]) ** 2)
        loss_pe22 = torch.mean(mask * (pred[:, 1] - target[:, 1]) ** 2)
        loss_pe33 = torch.mean(mask * (pred[:, 2] - target[:, 2]) ** 2)
        
        # Weighted sum
        total_loss = (self.alpha * loss_pe11 + 
                      self.beta * loss_pe22 + 
                      self.gamma * loss_pe33)
        
        return total_loss
    
    def get_component_losses(self, pred, target):
        """
        Return the independent losses for each component (used for monitoring)
        
        Returns:
            dict: {'pe11': loss_pe11, 'pe22': loss_pe22, 'pe33': loss_pe33}
        """
        mask = self._weight_mask_buffer.to(pred.device)
        
        loss_pe11 = torch.mean(mask * (pred[:, 0] - target[:, 0]) ** 2)
        loss_pe22 = torch.mean(mask * (pred[:, 1] - target[:, 1]) ** 2)
        loss_pe33 = torch.mean(mask * (pred[:, 2] - target[:, 2]) ** 2)
        
        return {
            'pe11': loss_pe11.item(),
            'pe22': loss_pe22.item(),
            'pe33': loss_pe33.item()
        }


class SimpleMSELoss(nn.Module):
    """
    Simple MSE Loss (used for comparative experiments)
    """
    def __init__(self):
        super(SimpleMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.mse(pred, target)


if __name__ == "__main__":
    # Test loss function
    print("=== Testing ComponentWeightedHAZLoss ===")
    
    # Create mock data
    batch_size = 2
    pred = torch.randn(batch_size, 3, 256, 256)
    target = torch.randn(batch_size, 3, 256, 256)
    
    # Test different weight configurations
    configs = [
        {"haz": 10, "alpha": 1.0, "beta": 1.0, "gamma": 1.0},
        {"haz": 20, "alpha": 1.0, "beta": 0.5, "gamma": 1.5},
        {"haz": 5, "alpha": 2.0, "beta": 1.0, "gamma": 0.5},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        criterion = ComponentWeightedHAZLoss(
            haz_multiplier=config['haz'],
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma']
        )
        
        loss = criterion(pred, target)
        component_losses = criterion.get_component_losses(pred, target)
        
        print(f"  Total loss: {loss.item():.6f}")
        print(f"  PE11 loss: {component_losses['pe11']:.6f}")
        print(f"  PE22 loss: {component_losses['pe22']:.6f}")
        print(f"  PE33 loss: {component_losses['pe33']:.6f}")
        
        assert loss.ndim == 0, "Loss should be a scalar"
    
    print("\n✓ All tests passed!")
