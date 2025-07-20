#the cell_id_to_celltype_lookup.csv is taken from the .TSV file which has the cell_ID on each slide and what cell type that matches to.
#IMPORTANT: the cell_IDs repeat each slide. As cell_ID 3 means different on slide T100 and T101.
import pandas as pd

# Read the existing lookup CSV file
lookup_df = pd.read_csv('cell_id_to_celltype_lookup.csv')

# Get unique celltypes in the order of appearance
unique_celltypes = lookup_df['celltype'].drop_duplicates().reset_index(drop=True)

# Create a mapping dictionary: celltype -> index (starting from 0)
celltype_to_index = {celltype: idx for idx, celltype in enumerate(unique_celltypes)}

# Add a new column 'celltype_index' by mapping the celltype column
lookup_df['celltype_index'] = lookup_df['celltype'].map(celltype_to_index)

# Save the updated lookup table to a new CSV file
lookup_df.to_csv('cell_id_to_celltype_lookup_with_index.csv', index=False)
print("Updated lookup table saved to cell_id_to_celltype_lookup_with_index.csv")
