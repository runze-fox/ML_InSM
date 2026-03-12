# TripleHead - Three-Head Independent Decoder Model

## Project Overview

TripleHead is an improved model architecture for predicting welding plastic strains. It adopts a **three-head independent upsampling** design philosophy, combined with **Optuna auto-tuning**, to improve the prediction accuracy of the three strain components: PE11, PE22, and PE33.

## Core Innovations

### 1. Three-Head Independent Decoder Architecture

```
Input Thickness T → Shared Backbone → ┬→ PE11 Decoder → PE11 Prediction
                                      ├→ PE22 Decoder → PE22 Prediction
                                      └→ PE33 Decoder → PE33 Prediction
```

**Design Concept**:
- **Geometric feature sharing**: T-joints of the same thickness have the same geometry.
- **Physical mapping independence**: The spatial distribution patterns of different strain components are different.

### 2. Component-Weighted HAZ Loss Function

$$Loss_{total} = \alpha \cdot Loss_{PE11} + \beta \cdot Loss_{PE22} + \gamma \cdot Loss_{PE33}$$

- Higher error weights in the HAZ (Heat-Affected Zone, near the weld).
- The loss weights for the three components can be adjusted independently.

### 3. Optuna Hyperparameter Optimization

Automatically searches 7 key parameters:
- `learning_rate`: [1e-5, 1e-2]
- `weight_decay`: [1e-6, 1e-3]
- `haz_multiplier`: [5, 50]
- `base_channels`: [32, 64, 128]
- `alpha`, `beta`, `gamma`: [0.5, 2.0]

## Project Structure

```
TripleHead/
├── data/
│   └── dataset.py               # Dataset class (supports unified normalization and individual thickness testing)
├── models/
│   ├── triple_head_model.py     # Three-head model architecture
│   └── losses.py                # Component-weighted loss functions
├── scripts/
│   ├── train_baseline.py        # Baseline training script
│   ├── optuna_tune.py           # Optuna auto-tuning script
│   ├── physical_continuity_test.py # Physical continuity testing and result visualization
│   ├── test.py                  # Single sample testing or early inference demonstration
│   └── model_summary.py         # Network structure summary printing
└── README.md
```

## Quick Start

### 1. Test Model Architecture

```bash
cd D:\CNN_InSM_New\TripleHead
conda activate ML

# Test model forward pass
python -c "from models.triple_head_model import TripleHeadPEModel; import torch; model = TripleHeadPEModel(64); print(model(torch.randn(2,1)).shape)"
```

### 2. Run Baseline Training

```bash
python scripts\train_baseline.py --epochs 100 --lr 0.001 --channels 128
```

### 3. Optuna Auto-Tuning

```bash
# Run Optuna hyperparameter search (default configurations included in the code)
python scripts\optuna_tune.py
```

### 4. Physical Continuity Test (Test Arbitrary Thickness)

Used to verify the model's physical continuity and prediction accuracy under different specified thicknesses:

```bash
python scripts\physical_continuity_test.py
```
> **Note**: You need to configure parameters such as `test_thicknesses` within the script. The test will generate `.npy` files of the predicted tensors for the corresponding thicknesses, as well as visual results showing an intuitive comparison between ground truth and predicted components.

## Key Features

✅ **Unified Normalization**: The training set and validation set use the same normalization parameters, supporting independent dynamic normalization for each component (PE11, PE22, PE33).
✅ **VRAM Optimization**: Automatically clears VRAM after each trial to prevent OOM crashes.
✅ **Auto-Pruning**: Terminates unpromising trials early to improve hyperparameter search efficiency.
✅ **Parameter Persistence**: Automatically saves the best hyperparameters as JSON, and saves the model to a specific results folder.

## Verification Workflow

1. **Unit Testing**: Verifies the model, loss functions, datasets, and auto-normalization logic.
2. **Baseline Training**: Confirms the model is trainable and converges.
3. **Optuna Tuning**: Searches for optimal hyperparameters and exports best model weights.
4. **Prediction Verification**: Uses `physical_continuity_test.py` to test the generalization ability on multiple interpolated thicknesses not in the training set.

## Expected Improvements

Compared to the original single-head model:
- PE11, PE22, PE33 each have independent feature extraction capabilities and decoding paths.
- Resolves the issue of magnitude differences between components by adjusting component weights.
- Optuna automatically finds optimal parameters such as learning rate and HAZ weights to enhance physical representation.

## Author

Runze Shi
