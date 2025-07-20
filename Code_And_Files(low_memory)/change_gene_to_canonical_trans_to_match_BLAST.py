#!/usr/bin/env python3
import pandas as pd
import anndata as ad

mouseSAM = ad.read_h5ad('/user/work/pr21872/Homologs/samap_directory/sam1_mouse.h5ad')
macaqueSAM = ad.read_h5ad('/user/work/pr21872/Homologs/samap_directory/sam2_macaque.h5ad')

mouse_gene_map = pd.read_csv('/user/work/pr21872/Homologs/samap_directory/sc_RNA_seq_CTXHPF_gene_IDs_Canonical_Trans.csv')
macaque_gene_map = pd.read_csv('/user/work/pr21872/Homologs/samap_directory/sn_RNA_seq_MacaqueGenesUsed_CanonicalTrans.csv')

# Create a mapping dictionary from Gene_stable_ID to Transcript_stable_ID
mouse_mapping_dict = dict(zip(mouse_gene_map['Transcript_stable_ID'], mouse_gene_map['Transcript_stable_ID_version']))
macaque_mapping_dict = dict(zip(macaque_gene_map['Transcript_stable_ID'], macaque_gene_map['Transcript_stable_ID_version']))

# Directly update adata.var_names with the mapped values
mouseSAM.var_names = mouseSAM.var_names.to_series().map(mouse_mapping_dict)
macaqueSAM.var_names = macaqueSAM.var_names.to_series().map(macaque_mapping_dict)

# Save the adata
mouseSAM.write_h5ad('/user/work/pr21872/Homologs/samap_directory/sam1_mouse.h5ad')
macaqueSAM.write_h5ad('/user/work/pr21872/Homologs/samap_directory/sam2_macaque.h5ad')

