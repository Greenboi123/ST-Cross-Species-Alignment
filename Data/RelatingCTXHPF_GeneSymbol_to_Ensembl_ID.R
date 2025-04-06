# Load necessary packages
library(biomaRt)
library(dplyr)

# Read the CSV file
df <- read.csv("D:\\Mouse\\Notebooks\\CTXHPF_gene.csv", row.names = 1, stringsAsFactors = FALSE)

# Extract gene symbols from the 'gene' column
genes <- df$gene

# Connect to the Ensembl BioMart for mouse
mart <- useMart("ensembl", dataset = "mmusculus_gene_ensembl")

# Get the Ensembl IDs for the provided gene symbols
results <- getBM(attributes = c("ensembl_gene_id", "external_gene_name"),
                 filters = "external_gene_name",
                 values = genes,
                 mart = mart)

# Merge the results with the original data frame while preserving the row order/index.
# Using left_join from dplyr which keeps the original row order.
merged_df <- left_join(df, results, by = c("gene" = "external_gene_name"))

# Remove rows where ensembl_gene_id is NA
merged_df <- merged_df[!is.na(merged_df$ensembl_gene_id), ]

# Subtract 1 from each row name (convert row names to numeric, subtract 1, then convert back to character)
new_rownames <- as.numeric(rownames(merged_df)) - 1
rownames(merged_df) <- new_rownames

# Save the merged mapping to a CSV file
write.csv(merged_df, file = "D:\\Mouse\\Notebooks\\mouse_entrez_to_ensembl_mapping_with_row_index.csv", row.names = TRUE)

