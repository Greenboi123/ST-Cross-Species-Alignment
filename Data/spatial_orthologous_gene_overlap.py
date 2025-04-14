import pandas as pd

mouse_genes_used = '/Users/felix/Downloads/MouseGenesUsedWithIDs.csv'
macaque_genes_used = '/Users/felix/Downloads/MacaqueGenesUsedWithIDs.csv'
all_mouse_macaque_121ortholog_overlap = '/Users/felix/Downloads/AllMacaqueFacsicus2Mouse121Orthologs.csv'

# Load CSV files into DataFrames
mouse_df = pd.read_csv(mouse_genes_used)
macaque_df = pd.read_csv(macaque_genes_used)
overlap_df = pd.read_csv(all_mouse_macaque_121ortholog_overlap)

# Filter overlap_df:
# - "Gene stable ID" should be in mouse_df['gene_identifier']
# - "Crab-eating macaque gene name" should be in macaque_df['gene_name']
filtered_overlap_df = overlap_df[
    overlap_df['Gene stable ID'].isin(mouse_df['gene_identifier']) &
    overlap_df['Crab-eating macaque gene name'].isin(macaque_df['gene_name'])
]

# Optionally, save the filtered DataFrame to a new CSV file
filtered_overlap_df.to_csv('/Users/felix/Downloads/filtered_overlap.csv', index=False)