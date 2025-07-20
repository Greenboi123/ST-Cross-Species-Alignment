# Load necessary package
library(biomaRt)

# Read the CSV file (adjust the file path as needed)
data <- read.csv("D:\\Macaque\\txtFiles\\mouse_VISp_gene_expression_matrices_2018-06-14\\mouse_VISp_2018-06-14_genes-rows.csv", stringsAsFactors = FALSE)

# Extract unique Entrez IDs from the column 'gene_entrez_id'
your_entrez_ids <- unique(data$gene_entrez_id)

# Set up connection to Ensembl BioMart for human genes
mart <- useMart("ensembl", dataset = "mmusculus_gene_ensembl")

# Retrieve mapping between Ensembl Gene IDs and Entrez Gene IDs
mapping <- getBM(attributes = c("ensembl_gene_id", "entrezgene_id"),
                 filters = "entrezgene_id",
                 values = your_entrez_ids,
                 mart = mart)

# Display the first few mappings
head(mapping)

write.csv(mapping, file="mouse_entrez_to_ensembl_mapping.csv", row.names=FALSE)
