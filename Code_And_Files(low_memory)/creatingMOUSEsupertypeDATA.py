#!/usr/bin/env python3
import pandas as pd
import anndata
import numpy as np


# metadata = pd.read_csv('/user/work/pr21872/SpaceData/bigData/Expressions/metadata/MERFISH-C57BL6J-638850-CCF/20231215/views/cell_metadata_with_parcellation_annotation.csv')
# metadata_isocortex = metadata[metadata['parcellation_division']=='Isocortex']
# del metadata
# adata = anndata.read_h5ad('/user/work/pr21872/SpaceData/bigData/Expressions/expression_matrices/MERFISH-C57BL6J-638850-imputed/20240831/C57BL6J-638850-imputed-log2.h5ad',backed='r')
# unique_supertypes = metadata_isocortex['supertype'].unique().to_list()
# unique_parcellations = metadata_isocortex['parcellation_structure'].unique().to_list()
# for supertype in unique_supertypes:
#     for parcellation in unique_parcellations:
#         supertype_in_parcellation = metadata_isocortex[metadata_isocortex['parcellation_structure']==parcellation]['supertype'].to_list()
#         supertypes_cellexpr = adata.obs_names.isin(supertype_in_parcellation)
#         supertype_parcellation_cells = adata[supertypes_cellexpr,:].to_memory()
#         # Total expression and Total counts across all supertypes_cellexpr for this supertype in this parcellation

# ---------------------------
# Load and filter metadata
# ---------------------------
metadata_file = '/user/work/pr21872/SpaceData/bigData/Expressions/metadata/MERFISH-C57BL6J-638850-CCF/20231215/views/cell_metadata_with_parcellation_annotation.csv'
metadata = pd.read_csv(metadata_file)

# Filter to only cells in the Isocortex division
metadata_isocortex = metadata[metadata['parcellation_division'] == 'Isocortex'].copy()
del metadata

# Set cell names to index
metadata_isocortex.set_index('cell_label', inplace=True)

# ---------------------------
# Load AnnData object
# ---------------------------
adata_file = '/user/work/pr21872/SpaceData/bigData/Expressions/expression_matrices/MERFISH-C57BL6J-638850-imputed/20240831/C57BL6J-638850-imputed-log2.h5ad'
adata = anndata.read_h5ad(adata_file, backed='r')

# ---------------------------
# Identify unique supertypes and parcellation structures
# ---------------------------
unique_supertypes = sorted(metadata_isocortex['supertype'].unique())       # Expected 104 isocortex cells
unique_parcellations = sorted(metadata_isocortex['parcellation_structure'].unique())  # Expected 43

n_supertypes = len(unique_supertypes)
n_parcellations = len(unique_parcellations)
n_genes = len(adata.var_names)

# ---------------------------
# Initialize output arrays
# ---------------------------
total_expression = np.zeros((n_supertypes, n_parcellations, n_genes), dtype=np.float32)
cell_count = np.zeros((n_supertypes, n_parcellations, n_genes), dtype=np.int32)
mean_expression = np.zeros((n_supertypes, n_parcellations, n_genes), dtype=np.float32)

# ---------------------------
# Loop over all (supertype, parcellation) combinations
# ---------------------------
for i, stype in enumerate(unique_supertypes):
    for j, parc in enumerate(unique_parcellations):
        # Identify cells in this combination based on supertype and parcellation
        cells = metadata_isocortex[
            (metadata_isocortex['supertype'] == stype) &
            (metadata_isocortex['parcellation_structure'] == parc)
        ].index
        
        # Ensure cells found in metadata match those in the AnnData object
        valid_cells = [cell for cell in cells if cell in adata.obs_names]
        
        if len(valid_cells) == 0:
            # Set mean expression to NaN when no cells are found
            mean_expression[i, j, :] = np.nan
            # Both total_expression and cell_count remain as 0 (or you could set them explicitly)
            continue

        # Subset the AnnData object using valid cells
        sub_adata = adata[valid_cells, :].to_memory()
        # Get the expression matrix 
        full_cells = sub_adata.X
        
        # Total expression: sum across all cells for each gene
        expr_sum = np.sum(full_cells, axis=0)
        total_expression[i, j, :] = expr_sum
        
        # Count total cells (even if a gene's expression is 0)
        n_cells = full_cells.shape[0]
        cell_count[i, j, :] = n_cells
        
        # Calculate mean expression: if there are cells, mean = total expression / count,
        # otherwise (n_cells==0) it was already handled above by setting to NaN.
        mean_expression[i, j, :] = expr_sum / n_cells

# ---------------------------
# Save the arrays into an NPZ file
# ---------------------------
output_file = '/user/work/pr21872/SpaceData/bigData/aggregate_data/mouse_aggregated_supertype_data.npz'
np.savez(output_file,
         total_expression=total_expression,
         cell_count=cell_count,
         mean_expression=mean_expression)

print(f"Saved expression data as {output_file}")




