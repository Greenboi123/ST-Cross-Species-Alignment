#!/usr/bin/env python3
import itertools
import pickle
import pandas as pd
import anndata
import numpy as np
import re
    
"""Gathering the aggregate data"""
mouse_aggregate_path = '/user/work/pr21872/SpaceData/bigData/aggregate_data/mouse_aggregated_supertype_data.npz'
macaque_aggreate_path = '/user/work/pr21872/SpaceData/bigData/aggregate_data/new_aggregated_data.npz'

"""Gather the mappings"""
cell_mappings_path = '/user/work/pr21872/Homologs/samap_directory/analyse_samap/mouseMacaqueCellmappingsAlign50.pkl'
with open(cell_mappings_path, 'rb') as f:
    cell_mappings_dict = pickle.load(f)

"""Gather the 121 orthologous genes"""
orthologs = pd.read_csv('/user/work/pr21872/test_aai/Encoder-Decoder/spatial_genes_overlap.csv')

"""Only keep the orthologous genes"""
# ------------- MOUSE -------------
# Gene order in .npz
adata_file = '/user/work/pr21872/SpaceData/bigData/Expressions/expression_matrices/MERFISH-C57BL6J-638850-imputed/20240831/C57BL6J-638850-imputed-log2.h5ad'
adata = anndata.read_h5ad(adata_file, backed='r')
mouse_gene_name_order = adata.var_names
# Gene othologs
mouse_orthologous_gene_names = orthologs["Gene stable ID"].to_list()
ortholog_set = set(mouse_orthologous_gene_names) #faster lookup
# .npz file
mouse_aggregate_data = np.load(mouse_aggregate_path)
mouse_mean_expression = mouse_aggregate_data['mean_expression'] #(296, 43, 8460)
# Only keep the genes in orthologs list.
mouse_gene_mask = np.array([gene in ortholog_set for gene in mouse_gene_name_order])
mouse_gene_indices_to_keep = np.where(mouse_gene_mask)[0]
filtered_mouse_mean_expression = mouse_mean_expression[:, :, mouse_gene_indices_to_keep].copy() #(296, 43, 4851)
# For space
del mouse_mean_expression, mouse_aggregate_data, adata

# ------------- Macaque -------------
# Gene order in .npz
macaque_aggregate_data = np.load(macaque_aggreate_path) # (15926, 141, 258)
macaque_genes = macaque_aggregate_data['genes'] # 15926 list
macaque_mean_expression = macaque_aggregate_data['mean']
macaque_cell_idx = macaque_aggregate_data['celltype_indices']
# Orthologous genes
macaque_orthologous_gene_names = orthologs["Crab-eating macaque gene name"].to_list()
macaque_ortholog_set = set(macaque_orthologous_gene_names) #faster lookup
# Only keep genes in orthologs list
macaque_gene_mask = np.array([gene in macaque_ortholog_set for gene in macaque_genes])
macaque_gene_indices_to_keep = np.where(macaque_gene_mask)[0]
filtered_macaque_mean_expression = macaque_mean_expression[macaque_gene_indices_to_keep, :, :].copy() # (4851, 141, 258)
# for space
del macaque_ortholog_set, macaque_mean_expression, macaque_aggregate_data

"""Reorganise so orthologous genes map to each others index"""
# Convert to a NumPy array for convenience
filtered_mouse_gene_names = np.array(mouse_gene_name_order)[mouse_gene_mask]
filtered_macaque_gene_names = np.array(macaque_genes)[macaque_gene_mask]

# For each gene in the canonical order, find where it appears in the filtered arrays.
mouse_otholog_gene_order_indices = []
macaque_otholog_gene_order_indices = []
for m_gene, maca_gene in zip(mouse_orthologous_gene_names, macaque_orthologous_gene_names):
    # Only include genes that exist in both filtered sets.
    if m_gene in filtered_mouse_gene_names and maca_gene in filtered_macaque_gene_names:
        # Find the index in the filtered mouse list.
        mouse_idx = np.where(filtered_mouse_gene_names == m_gene)[0][0]
        # Find the index in the filtered macaque list.
        macaque_idx = np.where(filtered_macaque_gene_names == maca_gene)[0][0]
        mouse_otholog_gene_order_indices.append(mouse_idx)
        macaque_otholog_gene_order_indices.append(macaque_idx)

# reorder the gene axes in your expression arrays so that gene i in mouse aligns with gene i in macaque.
final_filtered_mouse_mean_expression = filtered_mouse_mean_expression[:, :, mouse_otholog_gene_order_indices].copy()
final_filtered_macaque_mean_expression = filtered_macaque_mean_expression[macaque_otholog_gene_order_indices, :, :].copy()
del filtered_mouse_mean_expression, filtered_macaque_mean_expression
del mouse_orthologous_gene_names, macaque_orthologous_gene_names, mouse_gene_mask, macaque_gene_mask

"""Only keep the cells in the mappings"""
mouse_cells = list(cell_mappings_dict.keys())
macaque_cells = list(cell_mappings_dict.values())
# Get all unique cells from each list of cells and make it a list
macaque_cells = list(set([cell for cells in macaque_cells for cell in cells]))
# Remove prefix used for SAMap
mouse_cells = [name.replace("mm_", "", 1) if name.startswith("mm_") else name for name in mouse_cells]
macaque_cells = [name.replace("mq_", "", 1) if name.startswith("mq_") else name for name in macaque_cells]


# ------------- Macaque -------------
# Collect macaque metadata
macaque_metadata_file = '/user/work/pr21872/Homologs/rds2hda5/macaqueMetadata.csv'
macaque_metadata = pd.read_csv(macaque_metadata_file, usecols=['Plot','FullName'])
# Create a mapping from plot to fullname (used in the cell_mappings_dict)
plot_to_fullname = dict(zip(macaque_metadata['Plot'], macaque_metadata['FullName']))
# Collect indexs for order of cell types in sample and replace with names in mappings
macaque_cell_id_index_file = '/user/work/pr21872/test_aai/Encoder-Decoder/cell_id_to_celltype_lookup_with_index.csv'
macaque_cell_id_index = pd.read_csv(macaque_cell_id_index_file, usecols=['celltype','celltype_index'])
macaque_cell_id_index = macaque_cell_id_index.drop_duplicates()
macaque_cell_id_index = macaque_cell_id_index.sort_values('celltype_index')
# nan_celltype_indexes = macaque_cell_id_index.loc[macaque_cell_id_index['celltype'].isna(), 'celltype_index']
# Final order of cells in .npz file with their full name (replace nan with 'unknown'), it is a cell type with valid non-nans and non-zeros but don't know name...
macaque_cell_name_in_index = macaque_cell_name_in_index = [
    plot_to_fullname[plot] if pd.notna(plot) else 'unknown'
    for plot in macaque_cell_id_index['celltype']
]
# Filter for only cells used
macaque_cell_mask = np.array([cell in macaque_cells for cell in macaque_cell_name_in_index])
macaque_cell_indices_to_keep = np.where(macaque_cell_mask)[0]
# Store final ordering of cells
final_macaque_cell_names = [macaque_cell_name_in_index[i] for i in macaque_cell_indices_to_keep]
# Filter for only those cells
cells_filtered_macaque_mean_expression = final_filtered_macaque_mean_expression[:, :, macaque_cell_indices_to_keep].copy() # (4851, 141, <258)
# For space
del final_filtered_macaque_mean_expression

# ------------- MOUSE -------------
# Collect mouse cell types
mouse_metadata_file = '/user/work/pr21872/SpaceData/bigData/Expressions/metadata/MERFISH-C57BL6J-638850-CCF/20231215/views/cell_metadata_with_parcellation_annotation.csv'
mouse_metadata = pd.read_csv(mouse_metadata_file, usecols=['parcellation_division','supertype'])
# Filter to only cells in the Isocortex division
metadata_isocortex = mouse_metadata[mouse_metadata['parcellation_division'] == 'Isocortex'].copy()
# Order of cell types in sample
unique_supertypes = sorted(metadata_isocortex['supertype'].unique())
# Remove the numbering used by the mouse paper
unique_supertypes = [re.sub(r'^\d+\s*', '', name) for name in unique_supertypes]
# Filter for cells used in mappings
mouse_cell_mask = np.array([cell in mouse_cells for cell in unique_supertypes])
mouse_cell_indices_to_keep = np.where(mouse_cell_mask)[0]
# Store final cell names positions
final_mouse_cell_names = [unique_supertypes[i] for i in mouse_cell_indices_to_keep]
# Filter for only those cells
cells_filtered_mouse_mean_expression = final_filtered_mouse_mean_expression[mouse_cell_indices_to_keep, :, :].copy() # (<296, 43, 8460)
# For space 
del final_filtered_mouse_mean_expression

"""Calculate the indicies mappings for the cell type mappings from their cell names to their positions in the cells_filtered_mouse_mean_expression and cells_filtered_macaque_mean_expression"""
# Create lookup dictionaries: cell name -> index
mouse_cell_to_index = {cell: idx for idx, cell in enumerate(final_mouse_cell_names)}
macaque_cell_to_index = {cell: idx for idx, cell in enumerate(final_macaque_cell_names)}

# Save lookups:
with open('/user/work/pr21872/test_aai/Encoder-Decoder/mouse_celltype_to_idx.pkl', 'wb') as f:
    pickle.dump(mouse_cell_to_index, f)
with open('/user/work/pr21872/test_aai/Encoder-Decoder/macaque_celltype_to_idx.pkl', 'wb') as f:
    pickle.dump(macaque_cell_to_index, f)

# Replace cell names in cell_mappings_dict with their corresponding indices.
cell_mappings_indices = {}
for mouse_cell, macaque_cells in cell_mappings_dict.items():
    mouse_cell = mouse_cell.replace("mm_", "", 1)
    if mouse_cell in mouse_cell_to_index:
        mouse_idx = mouse_cell_to_index[mouse_cell]
        # Initialise the list
        cell_mappings_indices.setdefault(mouse_idx, [])
        # Loop through each macaque cell in the list 
        for macaque_cell in macaque_cells:
            macaque_cell = macaque_cell.replace("mq_", "", 1)
            if macaque_cell in macaque_cell_to_index:
                macaque_idx = macaque_cell_to_index[macaque_cell]
                cell_mappings_indices[mouse_idx].append(macaque_idx)
            else:
                print(f"Warning: Macaque cell {macaque_cell} not found in final macaque cell names.")
                # Three not found... But all part of many mappings so fine
                # Warning: Macaque cell NonNeuron ASC.17 A2M/IL18 not found in final macaque cell names.
                # Warning: Macaque cell NonNeuron ASC.18 IRX2/PAX3 not found in final macaque cell names.
                # Warning: Macaque cell GLU L2.10 USH1C/CDH22 not found in final macaque cell names.
    else:
        print(f"Warning: Mouse cell {mouse_cell} not found in final mouse cell names.")

# Save the mapping dictionary
with open('/user/work/pr21872/test_aai/Encoder-Decoder/cell_mapping_indicies.pkl', 'wb') as f:
    pickle.dump(cell_mappings_indices, f)
print(cell_mappings_indices)


"""Create the samples format"""
# ------------- Macaque -------------
macaque_mean_data_permuted = np.transpose(cells_filtered_macaque_mean_expression, (0, 2, 1)) # (4851, 141, <258) -> (4851, <258, 141)
print(macaque_mean_data_permuted.shape)
macaque_samples = macaque_mean_data_permuted.reshape(-1, macaque_mean_data_permuted.shape[2])
print("Macaque samples shape:", macaque_samples.shape)  # Expected: (4851*<258, 141)
np.save("/user/work/pr21872/test_aai/Encoder-Decoder/CellMapping_Macaque_Samples141Regions.npy", macaque_samples)

# ------------- MOUSE -------------
mouse_mean_data_permuted = np.transpose(cells_filtered_mouse_mean_expression, (2, 0, 1)) #(<296, 43, 4851) -> (4851, <296, 43)
print(mouse_mean_data_permuted.shape)
mouse_samples = mouse_mean_data_permuted.reshape(-1, mouse_mean_data_permuted.shape[2])
print("Mouse samples shape:", mouse_samples.shape)  # Expected: (4851*<296, 43)
np.save("/user/work/pr21872/test_aai/Encoder-Decoder/CellMapping_Mouse_Samples43Regions.npy", mouse_samples)

"""Save all look ups for the sample data that we have been using"""
# Full–data gene lookups:
# Mouse (8460)
mouse_full_gene_to_idx = {
    gene: idx
    for idx, gene in enumerate(mouse_gene_name_order)
}
# Macaque (15926)
macaque_full_gene_to_idx = {
    gene: idx
    for idx, gene in enumerate(macaque_genes)
}

# Mapping–only gene lookups:
mapping_mouse_genes = list(mouse_gene_name_order[mouse_gene_indices_to_keep][mouse_otholog_gene_order_indices])
mapping_macaque_genes = list(macaque_genes[macaque_gene_indices_to_keep][macaque_otholog_gene_order_indices])
mouse_mapping_gene_to_idx = {
    gene: idx
    for idx, gene in enumerate(mapping_mouse_genes)
}
macaque_mapping_gene_to_idx = {
    gene: idx
    for idx, gene in enumerate(mapping_macaque_genes)
}

# Full–data cell‑type lookups:
# Mouse 296
mouse_full_celltype_to_idx = {
    cell: idx
    for idx, cell in enumerate(unique_supertypes)
}
# Macaque 258
macaque_full_celltype_to_idx = {
    cell: idx
    for idx, cell in enumerate(macaque_cell_name_in_index)
}

# Persist all to disk:
lookups = {
    'mouse_full_gene_to_idx': mouse_full_gene_to_idx,
    'macaque_full_gene_to_idx': macaque_full_gene_to_idx,
    'mouse_mapping_gene_to_idx': mouse_mapping_gene_to_idx,
    'macaque_mapping_gene_to_idx': macaque_mapping_gene_to_idx,
    'mouse_full_celltype_to_idx': mouse_full_celltype_to_idx,
    'macaque_full_celltype_to_idx': macaque_full_celltype_to_idx,
}

for name, d in lookups.items():
    path = f'/user/work/pr21872/test_aai/Encoder-Decoder/{name}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(d, f)
