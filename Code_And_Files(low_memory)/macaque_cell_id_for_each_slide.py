# file CellID_Per_Slide_Mapping.pkl which has keys corresponding to the end of each file name such as T25 and T38, these keys return the dataframe for that file with 'cell_id' and 'celltype_index'

import pandas as pd

# Read the TSV file with the required columns.
cell_types = pd.read_csv(
    r"D:\Macaque\ST.CellID.CellType.3monkeys.all.tsv", 
    sep='\t', 
    usecols=['macaque', 'slide', 'cell_id', 'celltype']
)

cell_types_macaque1 = cell_types[cell_types['macaque'] == 'macaque1']

# Create a lookup dictionary where the key is the slide and 
# the value is the DataFrame of the corresponding cell_id and celltype rows.
slide_lookup = {
    slide: group[['cell_id', 'celltype']]
    for slide, group in cell_types_macaque1.groupby('slide')
}

# Example: Print the lookup for a specific slide.
example_slide = list(slide_lookup.keys())[0]
print("Slide:", example_slide)
print(slide_lookup[example_slide])


import pickle

with open('CellID_Per_Slide_Mapping.pkl', 'wb') as outfile:
    pickle.dump(slide_lookup, outfile)

print("Lookup saved as a pickle file.")