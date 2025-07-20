import pandas as pd

mousepaper = pd.read_csv(r"D:\Macaque\txtFiles\MousePaperClusters.csv")
clusters_used = (mousepaper['CTX.cluster_id'].unique()).tolist()

CTX_HPF_metadata = pd.read_csv(r"D:\Macaque\txtFiles\metadata.csv")
print(CTX_HPF_metadata.shape)
# Only include cluisters that link to the mouse paper clustes
CTX_HPF_metadata = CTX_HPF_metadata[CTX_HPF_metadata['cell_type_alias_id'].isin(clusters_used)]

print(CTX_HPF_metadata.shape)

CTX_HPF_metadata.to_csv(r'D:\Mouse\Notebooks\CTXHPF_Cells_Link_Mouse_Paper.csv',index=False)
