#!/usr/bin/env python3
import argparse
import os
import pickle
import anndata
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def get_sinusoidal_encoding(seq_len, embed_dim):
    pe = torch.zeros(seq_len, embed_dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class MultiTaskNumericEncoder(nn.Module):
    def __init__(self, seq_len=43, embed_dim=256, num_heads=4, num_layers=6, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.value_embedding = nn.Linear(1, embed_dim)
        pos_enc = get_sinusoidal_encoding(seq_len, embed_dim)
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_hidden):
        # x_hidden shape: (batch, seq_len)
        x_embed = self.value_embedding(x_hidden.unsqueeze(-1)) + self.pos_enc
        latent = self.transformer_encoder(x_embed)  # (batch, seq_len, embed_dim)
        return latent # Only need latent

# --- Decoder with the same architecture setup ---
class MultiTaskTranslationDecoder(nn.Module):
    def __init__(self, seq_len=141, embed_dim=256, num_heads=4, num_layers=6, dropout=0.1):
        """
        Note:
            In this design, the decoder does not use causal (masked) self-attention
            because the target is non-sequential. However, it still leverages
            the standard Transformer decoder layer to allow cross attention with
            the pretrained encoders latent representation.
        """
        super().__init__()
        self.seq_len = seq_len
        
        # Learnable query tokens that drive the decoder.
        self.decoder_query = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # Positional encoding for the decoder tokens.
        pos_enc = get_sinusoidal_encoding(seq_len, embed_dim)
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))  # shape: (1, seq_len, embed_dim)
        
        # Note: We are not passing any target mask so self-attention is unmasked.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # so we use (batch, seq, embed_dim) ordering
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Two heads for the multi-task outputs.
        self.classification_head = nn.Linear(embed_dim, 1)
        self.regression_head = nn.Linear(embed_dim, 1)
    
    def forward(self, encoder_latent):
        batch_size = encoder_latent.size(0)
        # Expand the learnable decoder query to match the batch size.
        # Add positional encodings; since the target data is non-sequential, these serve purely as additional features.
        decoder_input = self.decoder_query.expand(batch_size, -1, -1) + self.pos_enc
        
        # Pass through the Transformer decoder.
        # No mask is provided; cross-attention still operates between the decoder input and encoder's latent representation.
        decoder_output = self.transformer_decoder(decoder_input, encoder_latent)
        
        # Generate predictions for both tasks.
        classification_output = self.classification_head(decoder_output).squeeze(-1)
        regression_output = self.regression_head(decoder_output).squeeze(-1)
        
        return classification_output, regression_output
    
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
mouse_orthologous_gene_names = orthologs["Gene name"].to_list()
ortholog_set = set(mouse_orthologous_gene_names) #faster lookup
# .npz file
mouse_aggregate_data = np.load(mouse_aggregate_path)
mouse_mean_expression = mouse_aggregate_data['mean_expression'] #(104, 43, 8460)
# Only keep the genes in orthologs list.
mouse_gene_mask = np.array([gene in ortholog_set for gene in mouse_gene_name_order])
indices_to_keep = np.where(mouse_gene_mask)[0]
filtered_mouse_mean_expression = mouse_mean_expression[:, :, indices_to_keep].copy() #(104, 43, 4851)
# For space
del indices_to_keep, mouse_gene_mask, mouse_mean_expression, mouse_aggregate_data
del mouse_orthologous_gene_names, adata

# ------------- Macaque -------------
# Gene order in .npz
macaque_aggregate_data = np.load(macaque_aggreate_path) # (15926, 141, 258)
macaque_genes = macaque_aggregate_data['genes'] # 15926 list
macaque_mean_expression = macaque_aggregate_data['mean']
# Orthologous genes
macaque_orthologous_gene_names = orthologs["Crab-eating macaque gene name"].to_list()
macaque_ortholog_set = set(macaque_orthologous_gene_names) #faster lookup
# Only keep genes in orthologs list
macaque_gene_mask = np.array([gene in macaque_ortholog_set for gene in macaque_genes])
macaque_indices_to_keep = np.where(macaque_gene_mask)[0]
filtered_macaque_mean_expression = macaque_mean_expression[macaque_indices_to_keep, :, :].copy() # (4851, 141, 258)
# for space
del macaque_indices_to_keep, macaque_gene_mask, macaque_ortholog_set, macaque_orthologous_gene_names
del macaque_mean_expression, macaque_genes, macaque_aggregate_data

"""Reorganise so orthologous genes map to each others index"""
# Convert to a NumPy array for convenience
filtered_mouse_gene_names = np.array(mouse_gene_name_order)[mouse_gene_mask]
filtered_macaque_gene_names = np.array(macaque_genes)[macaque_gene_mask]

# For each gene in the canonical order, find where it appears in the filtered arrays.
mouse_order_indices = []
macaque_order_indices = []
for m_gene, maca_gene in zip(mouse_orthologous_gene_names, macaque_orthologous_gene_names):
    # Only include genes that exist in both filtered sets.
    if m_gene in filtered_mouse_gene_names and maca_gene in filtered_macaque_gene_names:
        # Find the index in the filtered mouse list.
        mouse_idx = np.where(filtered_mouse_gene_names == m_gene)[0][0]
        # Find the index in the filtered macaque list.
        macaque_idx = np.where(filtered_macaque_gene_names == maca_gene)[0][0]
        mouse_order_indices.append(mouse_idx)
        macaque_order_indices.append(macaque_idx)

# reorder the gene axes in your expression arrays so that gene i in mouse aligns with gene i in macaque.
final_filtered_mouse_mean_expression = filtered_mouse_mean_expression[:, :, mouse_order_indices].copy()
final_filtered_macaque_mean_expression = filtered_macaque_mean_expression[macaque_order_indices, :, :].copy()
del filtered_mouse_mean_expression, filtered_macaque_mean_expression

# verify the new shape or print a few gene names to confirm the alignment.
print("New mouse expression shape:", final_filtered_mouse_mean_expression.shape)
print("New macaque expression shape:", final_filtered_macaque_mean_expression.shape)
print("Aligned mouse genes:", filtered_mouse_gene_names[mouse_order_indices][:5])
print("Aligned macaque genes:", filtered_macaque_gene_names[macaque_order_indices][:5])

"""Only keep the cells in the mappings"""
mouse_cells = list(cell_mappings_dict.keys())
macaque_cells = list(dict.fromkeys(cell_mappings_dict.values()))
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
# Final order of cells in .npz file with their full name
macaque_cell_name_in_index = [plot_to_fullname[plot] for plot in macaque_cell_id_index['celltype']]
# Filter for only cells used
macaque_cell_mask = np.array([cell in macaque_cells for cell in macaque_cell_name_in_index])
macaque_indices_to_keep = np.where(macaque_cell_mask)[0]
# Store final ordering of cells
final_macaque_cell_names = [macaque_cell_name_in_index[i] for i in macaque_indices_to_keep]
# Filter for only those cells
cells_filtered_macaque_mean_expression = final_filtered_macaque_mean_expression[:, :, macaque_indices_to_keep].copy() # (4851, 141, <258)
# For space
del filtered_macaque_mean_expression

# ------------- MOUSE -------------
# Collect mouse cell types
mouse_metadata_file = '/user/work/pr21872/SpaceData/bigData/Expressions/metadata/MERFISH-C57BL6J-638850-CCF/20231215/views/cell_metadata_with_parcellation_annotation.csv'
mouse_metadata = pd.read_csv(mouse_metadata_file, usecols=['parcellation_division','supertype'])
# Filter to only cells in the Isocortex division
metadata_isocortex = mouse_metadata[mouse_metadata['parcellation_division'] == 'Isocortex'].copy()
# Order of cell types in sample
unique_supertypes = sorted(metadata_isocortex['supertype'].unique())
# Filter for cells used in mappings
mouse_cell_mask = np.array([cell in mouse_cells for cell in unique_supertypes])
mouse_indices_to_keep = np.where(mouse_cell_mask)[0]
# Store final cell names positions
final_mouse_cell_names = [unique_supertypes[i] for i in mouse_indices_to_keep]
# Filter for only those cells
cells_filtered_mouse_mean_expression = final_filtered_mouse_mean_expression[mouse_indices_to_keep, :, :].copy() # (<104, 43, 8460)
# For space 
del filtered_mouse_mean_expression

"""Calculate the indicies mappings for the cell type mappings from their cell names to their positions in the cells_filtered_mouse_mean_expression and cells_filtered_macaque_mean_expression"""
# Create lookup dictionaries: cell name -> index
mouse_cell_to_index = {cell: idx for idx, cell in enumerate(final_mouse_cell_names)}
macaque_cell_to_index = {cell: idx for idx, cell in enumerate(final_macaque_cell_names)}

# Replace cell names in cell_mappings_dict with their corresponding indices.
# Use a list for each mouse cell so that if there are multiple mappings they all are retained.
cell_mappings_indices = {}
for mouse_cell, macaque_cell in cell_mappings_dict.items():
    if mouse_cell in mouse_cell_to_index and macaque_cell in macaque_cell_to_index:
        mouse_idx = mouse_cell_to_index[mouse_cell]
        macaque_idx = macaque_cell_to_index[macaque_cell]
        # Append to list if key exists; otherwise, create a new list for that mouse index
        cell_mappings_indices.setdefault(mouse_idx, []).append(macaque_idx)
    else:
        print(f"Warning: Mapping {mouse_cell} -> {macaque_cell} not found in the final orders.")

# Save the mapping dictionary
with open('/user/work/pr21872/test_aai/Encoder-Decoder/cell_mapping_indicies.pkl', 'wb') as f:
    pickle.dump(cell_mappings_indices, f)

"""Create the samples format"""
# ------------- Macaque -------------
macaque_mean_data_permuted = np.transpose(cells_filtered_macaque_mean_expression, (0, 2, 1)) # (4851, 141, <258) -> (4851, <258, 141)
macaque_samples = macaque_mean_data_permuted.reshape(-1, macaque_mean_data_permuted.shape[1])
print("Macaque samples shape:", macaque_samples.shape)  # Expected: (4851*<258, 141)
np.save("/user/work/pr21872/test_aai/Encoder-Decoder/CellMapping_Macaque_Samples141Regions.npy", macaque_samples)

# ------------- MOUSE -------------
mouse_mean_data_permuted = np.transpose(cells_filtered_mouse_mean_expression, (2, 1, 0)) #(<104, 43, 4851) -> (4851, <104, 43)
mouse_samples = mouse_mean_data_permuted.reshape(-1, mouse_mean_data_permuted.shape[1])
print("Mouse samples shape:", mouse_samples.shape)  # Expected: (4851*<104, 43)
np.save("/user/work/pr21872/test_aai/Encoder-Decoder/CellMapping_Mouse_Samples43Regions.npy", mouse_samples)


