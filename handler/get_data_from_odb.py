# # -*- coding: mbcs -*-
# from odbAccess import *
# from abaqusConstants import *
# import os
# 
# # 1. Set Gauss point constant (1/sqrt(3))
# GP = 0.577350269189626
# # Local coordinates corresponding to integration points 1-8 of C3D8T element
# GAUSS_LOCS = [
#     (-GP, -GP, -GP), (GP, -GP, -GP), (GP, GP, -GP), (-GP, GP, -GP),
#     (-GP, -GP,  GP), (GP, -GP,  GP), (GP, GP,  GP), (-GP, GP,  GP)
# ]
# 
# def get_Ni_values(xi, eta, zeta):
#     """3D linear hexahedral shape functions """
#     return [
#         0.125 * (1 - xi) * (1 - eta) * (1 - zeta),
#         0.125 * (1 + xi) * (1 - eta) * (1 - zeta),
#         0.125 * (1 + xi) * (1 + eta) * (1 - zeta),
#         0.125 * (1 - xi) * (1 + eta) * (1 - zeta),
#         0.125 * (1 - xi) * (1 - eta) * (1 + zeta),
#         0.125 * (1 + xi) * (1 - eta) * (1 + zeta),
#         0.125 * (1 + xi) * (1 + eta) * (1 + zeta),
#         0.125 * (1 - xi) * (1 + eta) * (1 + zeta)
#     ]
# 
# # =========================== Configuration Area ===========================
# odb_files = ['TJoint_T_10mm.odb', 'TJoint_T_12mm.odb', 
#              'TJoint_T_14mm.odb', 'TJoint_T_16mm.odb', 'TJoint_T_18mm.odb']
# instance_name = 'Part-1-1'
# output_dir = r'F:\RS_Abaqus_Python\postProcessing'
# 
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# 
# # ===============================================================
# 
# for odb_name in odb_files:
#     print "Processing ODB: " + odb_name
#     try:
#         odb = openOdb(odb_name)
#     except:
#         print "Error: Cannot open " + odb_name
#         continue
# 
#     # Using .upper() as per your original requirement
#     inst = odb.rootAssembly.instances[instance_name.upper()]
#     nodes = inst.nodes
#     last_frame = odb.steps.values()[-1].frames[-1]
#     
#     # =========================== Fixed Configuration ===========================
#     # Since mesh_size_z is 0.005 (5mm)
#     # We set tolerance to 0.003, which ensures we only capture a single row of elements (the row where Centroid is closest to target_z)
#     mesh_size_z = 0.005 
#     tolerance = 0.003
#     # Z-axis midpoint calculation logic remains unchanged
#     z_coords = [n.coordinates[2] for n in nodes]
#     z_min, z_max = min(z_coords), max(z_coords)
#     target_z = (z_min + z_max) / 2.0
# 
#     print "Target Z-mid: %.4f, Using narrow tolerance: %.4f" % (target_z, tolerance)
#     
#     output_csv = os.path.join(output_dir, 'FIXED_INTEGPT_PE_' + odb_name.replace('.odb', '.csv'))
#     
#     pe_field = last_frame.fieldOutputs['PE'].getSubset(position=INTEGRATION_POINT)
# 
#     with open(output_csv, 'wb') as f:
#         f.write('FileName,ElementLabel,IP_Index,X,Y,Z,PE11,PE22,PE33\n')
# 
#                 # ... Enter loop ...
#         for val in pe_field.values:
#             # Get coordinates and calculate z_center per your original logic
#             elem = inst.elements[val.elementLabel-1]
#             node_indices = elem.connectivity
#             try:
#                 elem_nodes_coords = [nodes[i-1].coordinates for i in node_indices]
#                 # Adopt sum([...]) format that you verified
#                 z_center = sum([c[2] for c in elem_nodes_coords]) / len(node_indices)
#             except:
#                 continue
#             
#             # Only enter calculation if element centroid is extremely close to target_z
#             if abs(z_center - target_z) < tolerance:
#                 # --- Method B: Accurate coordinate calculation ---
#                 ip_idx = val.integrationPoint
#                 xi, eta, zeta = GAUSS_LOCS[ip_idx - 1]
#                 Ni = get_Ni_values(xi, eta, zeta)
#                 
#                 px = sum([Ni[j] * elem_nodes_coords[j][0] for j in range(8)])
#                 py = sum([Ni[j] * elem_nodes_coords[j][1] for j in range(8)])
#                 pz = sum([Ni[j] * elem_nodes_coords[j][2] for j in range(8)])
# 
#                 # [Core Improvement]: Narrow down the range again before writing to CSV
#                 # Because an element has 8 integration points, 4 in Z+ direction, 4 in Z- direction
#                 # If you only want "one" plane, you can just keep the 4 points closest to target_z
#                 # If you want to keep all info from this entire row of elements for calculation, then keep all 8 points
#                 
#                 pe11, pe22, pe33 = val.data[0], val.data[1], val.data[2]
#                 f.write('%s,%d,%d,%.6f,%.6f,%.6f,%.8e,%.8e,%.8e\n' % (
#                     odb_name, val.elementLabel, ip_idx, px, py, pz, pe11, pe22, pe33))
#         
#         # for val in pe_field.values:
#         #     elem = inst.elements[val.elementLabel-1]
#         #     node_indices = elem.connectivity
#             
#         #     # Use list comprehension consistent with your original script to calculate z_center, avoiding TypeError
#         #     try:
#         #         elem_nodes_coords = [nodes[i-1].coordinates for i in node_indices]
#         #         z_center = sum([c[2] for c in elem_nodes_coords]) / len(node_indices)
#         #     except:
#         #         continue
#                 
#         #     if abs(z_center - target_z) < tolerance:
#         #         # --- Method B: Accurate coordinate calculation ---
#         #         ip_idx = val.integrationPoint
#         #         xi, eta, zeta = GAUSS_LOCS[ip_idx - 1]
#         #         Ni = get_Ni_values(xi, eta, zeta)
#                 
#         #         # Sum using list comprehension as well
#         #         px = sum([Ni[j] * elem_nodes_coords[j][0] for j in range(8)])
#         #         py = sum([Ni[j] * elem_nodes_coords[j][1] for j in range(8)])
#         #         pz = sum([Ni[j] * elem_nodes_coords[j][2] for j in range(8)])
#                 
#         #         pe11, pe22, pe33 = val.data[0], val.data[1], val.data[2]
#                 
#         #         f.write('%s,%d,%d,%.6f,%.6f,%.6f,%.8e,%.8e,%.8e\n' % (
#         #             odb_name, val.elementLabel, ip_idx, px, py, pz, pe11, pe22, pe33))
# 
#     odb.close()
#     print "Finished: " + output_csv

# -*- coding: mbcs -*-
from odbAccess import *
from abaqusConstants import *
import os

# 1. Set Gauss point constant (1/sqrt(3))
GP = 0.577350269189626
GAUSS_LOCS = [
    (-GP, -GP, -GP), (GP, -GP, -GP), (GP, GP, -GP), (-GP, GP, -GP),
    (-GP, -GP,  GP), (GP, -GP,  GP), (GP, GP,  GP), (-GP, GP,  GP)
]

def get_Ni_values(xi, eta, zeta):
    """3D linear hexahedral shape functions"""
    return [
        0.125 * (1 - xi) * (1 - eta) * (1 - zeta),
        0.125 * (1 + xi) * (1 - eta) * (1 - zeta),
        0.125 * (1 + xi) * (1 + eta) * (1 - zeta),
        0.125 * (1 - xi) * (1 + eta) * (1 - zeta),
        0.125 * (1 - xi) * (1 - eta) * (1 + zeta),
        0.125 * (1 + xi) * (1 - eta) * (1 + zeta),
        0.125 * (1 + xi) * (1 + eta) * (1 + zeta),
        0.125 * (1 - xi) * (1 + eta) * (1 + zeta)
    ]

# =========================== Configuration Area ===========================

# odb_files = ['TJoint_T_15p0mm.odb','TJoint_T_15p5mm.odb','TJoint_T_16p0mm.odb','TJoint_T_16p5mm.odb', 
# 'TJoint_T_17p0mm.odb','TJoint_T_17p5mm.odb','TJoint_T_18p0mm.odb','TJoint_T_18p5mm.odb','TJoint_T_19p0mm.odb',
# 'TJoint_T_19p5mm.odb','TJoint_T_20p0mm.odb','TJoint_T_20p5mm.odb','TJoint_T_21p0mm.odb','TJoint_T_21p5mm.odb',
# 'TJoint_T_22p0mm.odb','TJoint_T_22p5mm.odb','TJoint_T_23p0mm.odb','TJoint_T_23p5mm.odb','TJoint_T_24p0mm.odb',
# 'TJoint_T_24p5mm.odb','TJoint_T_25p0mm.odb','TJoint_T_25p5mm.odb']
odb_files = ['TJoint_T_23p5mm.odb', 'TJoint_T_24p5mm.odb','TJoint_T_25p5mm.odb']

instance_name = 'Part-1-1'
mesh_size_z = 0.005  # 5mm mesh
output_dir = r'D:\CNN_InSM_New\data_input\abaqus_data\Web8mm' 
# ===============================================================

for odb_name in odb_files:
    print "Processing: " + odb_name
    try:
        odb = openOdb(odb_name)
    except:
        continue

    inst = odb.rootAssembly.instances[instance_name.upper()]
    nodes = inst.nodes
    last_frame = odb.steps.values()[-1].frames[-1]
    
    # Auto-locate midpoint
    z_coords = [n.coordinates[2] for n in nodes]
    target_z = (min(z_coords) + max(z_coords)) / 2.0
    
    # Set slicing filter window: only fetch the first IP layer behind the midpoint
    # Logic: pz must be greater than midpoint, and strictly smaller than midpoint + half mesh width
    z_lower_bound = target_z
    z_upper_bound = target_z + (0.5 * mesh_size_z)
    
    output_csv = os.path.join(output_dir, 'PURE_SLICE_' + odb_name.replace('.odb', '.csv'))
    pe_field = last_frame.fieldOutputs['PE'].getSubset(position=INTEGRATION_POINT)

    with open(output_csv, 'wb') as f:
        f.write('FileName,ElementLabel,IP_Index,X,Y,Z,PE11,PE22,PE33\n')
        
        for val in pe_field.values:
            elem = inst.elements[val.elementLabel-1]
            node_indices = elem.connectivity
            
            try:
                # Pre-fetch node coordinates
                elem_nodes_coords = [nodes[i-1].coordinates for i in node_indices]
            except:
                continue
                
            # Calculate exact IP coordinates
            ip_idx = val.integrationPoint
            xi, eta, zeta = GAUSS_LOCS[ip_idx - 1]
            Ni = get_Ni_values(xi, eta, zeta)
            
            pz = sum([Ni[j] * elem_nodes_coords[j][2] for j in range(8)])

            # [Core Logic Correction]: Force retaining strictly one unique Z-plane data
            if z_lower_bound < pz < z_upper_bound:
                px = sum([Ni[j] * elem_nodes_coords[j][0] for j in range(8)])
                py = sum([Ni[j] * elem_nodes_coords[j][1] for j in range(8)])
                
                pe11, pe22, pe33 = val.data[0], val.data[1], val.data[2]
                f.write('%s,%d,%d,%.6f,%.6f,%.6f,%.8e,%.8e,%.8e\n' % (
                    odb_name, val.elementLabel, ip_idx, px, py, pz, pe11, pe22, pe33))

    odb.close()
    print "Pure single slice saved: " + output_csv