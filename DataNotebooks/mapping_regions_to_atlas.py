import pandas as pd
import json

# Load your region mapping from JSON file
with open('region_mapping.json', 'r') as infile:
    region_mapping = json.load(infile)

region_atlas_df = pd.read_csv('Macaque_region_to_atlas_mappings.csv')

# Prepare a list for the flattened mapping.
flat_mapping = []

# Iterate over each region and its list of global_region_ids.
for region, mapping in region_mapping.items():
    # Look up the atlas number for this region.
    atlas_row = region_atlas_df[region_atlas_df['region'] == region]
    if not atlas_row.empty:
        atlas_number = atlas_row['atlas_number'].values[0]
    else:
        atlas_number = None  # or handle missing atlas number as needed
    # Add a record for each global_region_id in the region.
    for global_id in mapping['global_region_id']:
        flat_mapping.append({
            'global_region_id': global_id,
            'region': region,
            'atlas_number': atlas_number
        })

# Convert the flat mapping to a DataFrame.
lookup_df = pd.DataFrame(flat_mapping)

# Remove duplicate rows, if any.
lookup_df = lookup_df.drop_duplicates()

# Save the unique lookup table to a CSV file.
lookup_df.to_csv('global_region_id_to_atlas_number.csv', index=False)
print("Unique lookup table saved to global_region_id_to_atlas_number.csv")
