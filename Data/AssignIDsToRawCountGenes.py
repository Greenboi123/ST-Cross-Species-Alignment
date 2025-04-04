import os

# Change to your desired directory
os.chdir("D:\Macaque")  # Replace with your actual path


import csv

def load_gene_mapping(genes_info_file):
    """Load gene ID and gene name mappings from the reference file."""
    gene_id_to_name = {}
    gene_name_to_id = {}

    with open(genes_info_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2:
                gene_id, gene_name = row
                gene_id_to_name[gene_id] = gene_name
                gene_name_to_id[gene_name] = gene_id

    return gene_id_to_name, gene_name_to_id

def update_query_genes(query_file, genes_info_file, output_file):
    """Match gene IDs and names in the query file and write updated results."""
    gene_id_to_name, gene_name_to_id = load_gene_mapping(genes_info_file)

    with open(query_file, 'r', encoding='utf-8') as file:
        query_genes = [line.strip() for line in file if line.strip()]

    updated_results = []
    
    for gene in query_genes:
        if gene in gene_id_to_name:  # If it's a gene ID
            updated_results.append(f"{gene},{gene_id_to_name[gene]}")
        elif gene in gene_name_to_id:  # If it's a gene name
            updated_results.append(f"{gene_name_to_id[gene]},{gene}")
        else:
            updated_results.append(f"{gene},Not Found")

    # Save results
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("\n".join(updated_results) + "\n")

    print(f"Updated file saved as {output_file}")

# Example usage
query_file = "D:\Macaque\MacaqueGenesUsed.txt"  # Replace with your actual file
genes_info_file = "D:\Macaque\MacaqueGenesFromEnsembl.txt"  # Replace with your actual file
output_file = "updated_query_genes.txt"

update_query_genes(query_file, genes_info_file, output_file)

