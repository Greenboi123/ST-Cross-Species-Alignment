import pandas as pd

# Read the CSV file (replace 'your_file.csv' with your actual filename)
df = pd.read_csv('regions-macaque1.csv')

# Helper function to extract the region from the origin_name
def extract_region(origin_name):
    parts = origin_name.split('-')
    return parts[1] if len(parts) >= 3 else None

# Create a new column 'region' by extracting it from 'origin_name'
df['region'] = df['origin_name'].apply(extract_region)

# List of valid regions
regions_list = [
    "1/2", "10mc", "10mr", "10o", "11l", "11m", "12l", "12m", "12o", "12r", 
    "13a", "13b", "13l", "13m", "14c", "14r", "23a", "23b", "23c", "24a", "24a'", 
    "24b", "24b'", "24c", "24c'", "25", "29", "30", "31", "32", "35", "36c", "36p", 
    "36r", "3a/b", "44", "45a", "45b", "46d", "46v", "7op", "8Ad", "8Av", "8Bd", 
    "8Bm", "8Bs", "9d", "9m", "AI", "AIP", "AL", "CL", "CM", "CPB", "DP", "EC", 
    "ECL", "EI", "ELc", "ELr", "EO", "ER", "F1", "F2", "F3", "F4", "F5", "F6", "F7", 
    "FST", "G", "IPa", "Ia", "Iai", "Ial", "Iam", "Iapl", "Iapm", "Id", "Ig", "LIPd", 
    "LIPv", "LOP", "MIP", "ML", "MST", "MT", "Opt", "PE", "PEa", "PEc", "PEci", "PF", 
    "PFG", "PG", "PGa", "PGm", "PIP", "Pi", "Pir", "PrCO", "R", "RM", "RPB", "RT", 
    "RTL", "RTM", "RTp", "Ri", "SII", "STGr", "TAa", "TEO", "TEa", "TEad", "TEav", 
    "TEm", "TEpd", "TEpv", "TF", "TFO", "TGa", "TGdd", "TGdg", "TGsts", "TGvd", 
    "TGvg", "TH", "TPO", "Tpt", "V1", "V2", "V3A", "V3d", "V3v", "V4", "V4t", "V4v", 
    "V6", "V6Ad", "V6Av", "VIP", "v23b"
]

# Filter the DataFrame to only include rows with valid regions
df = df[df['region'].isin(regions_list)]

# Group by region and aggregate both the 'global_region_id' and 'chip' columns into lists
region_mapping = df.groupby('region').agg({
    'global_region_id': list,
    'chip': list
}).to_dict(orient='index')

import json

# Save the region_mapping dictionary to a JSON file
with open('region_mapping.json', 'w') as outfile:
    json.dump(region_mapping, outfile, indent=4)

print("Region mapping saved to region_mapping.json")
