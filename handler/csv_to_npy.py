import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# # 1. Load the corrected data
# # Make sure the filename matches what you generated locally
# file_path = r'D:\CNN_InSM_New\data_input\abaqus_data\Web8mm\Plate8p0mm.csv' 
# df = pd.read_csv(file_path)

# # 2. Create 3D canvas
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # 3. Plot scatter diagram
# # X: Transverse, Y: Thickness direction, Z: Welding direction
# # c=df['PE11'] colors based on plastic strain magnitude
# img = ax.scatter(df['X'], df['Z'], df['Y'], c=df['PE11'], cmap='jet', s=10, alpha=0.6)

# # 4. Set axis labels (consistent with your Abaqus coordinate system)
# ax.set_xlabel('X (Transverse)')
# ax.set_ylabel('Z (Welding Direction)')
# ax.set_zlabel('Y (Thickness)')
# ax.set_title('3D Distribution of Integration Points (Method B)')

# # Add colorbar
# fig.colorbar(img, ax=ax, label='PE11 Intensity')

# # 5. Perspective optimization: To clearly see the stratification in the thickness direction, tilt it slightly
# ax.view_init(elev=20, azim=-45)

# print("Generating 3D view...")
# plt.show()

# # --- Local zoom-in check ---
# # Select the first 2 elements (8 points each) and print coordinates to check if there are differences in Y
# print("\n--- Local coordinate check (Checking first 2 elements) ---")
# check_df = df.head(16) # 2 elements * 8 integration points
# print(check_df[['ElementLabel', 'IP_Index', 'X', 'Y', 'PE11']])

# import pandas as pd
# import numpy as np
# from scipy.interpolate import griddata
# import matplotlib.pyplot as plt
# import os

# # --- Configuration Area ---
# input_dir = r'F:\InSM_python\CNN-InSM\postProcessing'
# output_dir = r'F:\InSM_python\CNN-InSM\CNN_Dataset'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Define the globally uniform grid boundary (based on the actual maximum dimensions of your T-joints)
# # Make sure all samples (10mm to 18mm) can fall entirely into this range
# X_MIN, X_MAX = -0.02, 0.02    # Assumed transverse bounds
# Y_MIN, Y_MAX = -0.02, 0.02    # Assumed thickness direction bounds
# GRID_RES = 512                # CNN input resolution

# # --- Processing Logic ---
# def process_to_images():
#     csv_files = [f for f in os.listdir(input_dir) if f.startswith('PURE_SLICE')]
    
#     # Define standard grid
#     xi = np.linspace(X_MIN, X_MAX, GRID_RES)
#     yi = np.linspace(Y_MIN, Y_MAX, GRID_RES)
#     xi, yi = np.meshgrid(xi, yi)

#     for csv_file in csv_files:
#         print(f"Converting {csv_file} to tensor...")
#         path = os.path.join(input_dir, csv_file)
#         df = pd.read_csv(path)

#         points = df[['X', 'Y']].values
        
#         # Extract three plastic strain components simultaneously
#         channels = []
#         for component in ['PE11', 'PE22', 'PE33']:
#             values = df[component].values
#             # Map the scatter points to the grid using linear interpolation
#             grid_data = griddata(points, values, (xi, yi), method='linear', fill_value=0)
#             channels.append(grid_data)
        
#         # Stack into a (128, 128, 3) matrix
#         image_tensor = np.stack(channels, axis=-1)
        
#         # Save as numpy format for easy consumption by deep learning
#         save_name = csv_file.replace('.csv', '.npy')
#         np.save(os.path.join(output_dir, save_name), image_tensor)
        
#         # Visualize one of the components (PE11) for checks
#         if '10mm' in csv_file:
#             plt.figure(figsize=(8, 4))
#             plt.imshow(image_tensor[:,:,0], extent=(X_MIN, X_MAX, Y_MIN, Y_MAX), 
#                        origin='lower', cmap='jet')
#             plt.colorbar(label='PE11')
#             plt.title(f'Gridded Profile ({GRID_RES}x{GRID_RES}) - {csv_file}')
#             plt.savefig(os.path.join(output_dir, csv_file.replace('.csv', '.png')))
#             plt.close()

#     print(f"Preprocessing completed! All samples saved to: {output_dir}")

# if __name__ == "__main__":
#     process_to_images()


#--Update on Jan 20th
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter # Used for Gaussian smoothing
import matplotlib.pyplot as plt
import os

# --- 1. Configuration Area ---
input_dir = r'D:\CNN_InSM_New\data_input\abaqus_data\Web15mm'
output_dir = r'D:\CNN_InSM_New\data_input\numpy_data\from_interpo\Web15mm'

# Gaussian smoothing parameter (sigma)
SIGMA = 0.3

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Selected best resolution and ROI boundaries
GRID_RES = 256                
X_MIN, X_MAX = -0.02, 0.02    # Fixed 40mm square region
Y_MIN, Y_MAX = -0.02, 0.02    

# T-joint geometric parameters
W_WEB = 0.008      # Web width: can vary with model, e.g., 9mm, 10mm
L_LEG = 0.006       # Weld leg length is 6mm

# --- 2. Geometry Mask Logic ---
def get_geometry_mask(xi, yi, t_f, w_w, l_l):
    """
    Physical mask logic based on the origin being at the top surface of the base plate
    """
    mask = np.zeros_like(xi, dtype=bool)
    # 1. Base plate (from -t_f down to 0) -> wait, actually from -t_f to 0 on the Y axis
    mask |= ((yi >= -t_f) & (yi <= 0))
    # 2. Web plate (y > 0 region)
    mask |= ((yi > 0) & (np.abs(xi) <= w_w / 2))
    # 3. Weld triangles
    mask |= (xi >= w_w/2) & (xi <= w_w/2 + l_l) & (yi >= 0) & (yi <= -xi + (w_w/2 + l_l))
    mask |= (xi <= -w_w/2) & (xi >= -w_w/2 - l_l) & (yi >= 0) & (yi <= xi + (w_w/2 + l_l))
    return mask

# --- 3. Core Processing Logic ---
def process_to_dataset():
    csv_files = [f for f in os.listdir(input_dir) if f.startswith('Plate')]
    
    # Establish standard grid
    xi_vec = np.linspace(X_MIN, X_MAX, GRID_RES)
    yi_vec = np.linspace(Y_MIN, Y_MAX, GRID_RES)
    xi, yi = np.meshgrid(xi_vec, yi_vec)

    for csv_file in csv_files:
        try:
            # Dynamically fetch thickness param (Supports '19p0' formatting, where 'p' implies a decimal point)
            # The filename is currently like 'Plate8p0mm.csv'
            thickness_str = csv_file.replace('Plate', '').replace('.csv', '').replace('mm', '')
            thickness_str = thickness_str.replace('p', '.')  # Replace 'p' with decimal point
            current_t = float(thickness_str) / 1000.0
        except:
            continue

        print(f"Generating Dataset: {csv_file} | T = {current_t*1000}mm")
        path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(path)
        points = df[['X', 'Y']].values
        
        # Pre-generate geometry physical mask
        mask = get_geometry_mask(xi, yi, current_t, W_WEB, L_LEG)
        
        channels = []
        # Process the three physical channels: PE11, PE22, PE33
        for component in ['PE11', 'PE22', 'PE33']:
            values = df[component].values
            
            # A. Linear interpolation to capture the gradients
            grid_data = griddata(points, values, (xi, yi), method='linear', fill_value=0)
            grid_data = np.nan_to_num(grid_data)
            
            # B. Apply normalized Gaussian smoothing convolution
            # This step simulates natural physical continuity and removes meshing noise
            grid_data = gaussian_filter(grid_data, sigma=SIGMA)
            
            # C. Physical cropping: constraining to geometric boundaries
            grid_data = grid_data * mask 
            
            channels.append(grid_data)
        
        # Stack into (256, 256, 3) tensor and save the raw numeric values
        image_tensor = np.stack(channels, axis=-1)
        
        # Standardize output filenames: replace 'p' with '.'
        standardized_name = csv_file.replace('p', '.')
        np.save(os.path.join(output_dir, standardized_name.replace('.csv', '.npy')), image_tensor)
        
        # Save a preview figure for quality validation
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mask, extent=(X_MIN, X_MAX, Y_MIN, Y_MAX), origin='lower', cmap='gray')
        plt.axhline(0, color='blue', linestyle=':', label='Top Surface')
        plt.title(f"Geometry Mask (T={current_t*1000}mm)")
        
        plt.subplot(1, 2, 2)
        plt.imshow(image_tensor[:,:,0], extent=(X_MIN, X_MAX, Y_MIN, Y_MAX), origin='lower', cmap='jet')
        plt.colorbar(label='PE11 Raw Value')
        plt.title("Optimized PE Profile")
        plt.savefig(os.path.join(output_dir, standardized_name.replace('.csv', '.png')))
        plt.close()

    print(f"All datasets successfully regenerated and saved to: {output_dir} (sigma={SIGMA})")

if __name__ == "__main__":
    process_to_dataset()