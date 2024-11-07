import pandas as pd
import numpy as np

# Load the CSV file into a DataFrame
df = pd.read_csv(r'C:/Users/fjpgr/Downloads/Technical Project/metadata.csv')

# # Define the output file path
# output_file_path = r'C:/Users/fjpgr/Downloads/Technical Project/metadata_overview.txt'

# # Open the file for writing
# with open(output_file_path, 'w') as f:
#     for column in df.columns:
#         f.write(f"Column: {column}\n")
#         f.write(f"Data Type: {df[column].dtype}\n")
#         f.write("First 5 values:\n")
#         f.write("\n".join(map(str, df[column].head(5).to_list())) + "\n\n")

# print(f"Metadata overview saved to {output_file_path}")

print(np.unique(df['subregion']))