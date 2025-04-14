#!/usr/bin/env python3
from samap.mapping import SAMAP
from samap.analysis import (get_mapping_scores, GenePairFinder,
                            sankey_plot, chord_plot, CellTypeTriangles, 
                            ParalogSubstitutions, FunctionalEnrichment,
                            convert_eggnog_to_homologs, GeneTriangles)
from samalg import SAM
import pandas as pd
import pickle
import anndata

# A=pd.read_csv('/user/work/pr21872/Homologs/mapsOriginal/mqmm/mm_to_mq.txt',sep='\t',index_col=0,header=None)
# B=pd.read_csv('/user/work/pr21872/Homologs/mapsOriginal/mqmm/mq_to_mm.txt',sep='\t',index_col=0,header=None)

# # fn1 = '/user/work/pr21872/Homologs/Cleaning_stratisfied/CTX_Strat_100k.h5ad'
# # fn2 = '/user/work/pr21872/Homologs/Cleaning_stratisfied/macaque_Strat_100k.h5ad'
fn1 = '/user/work/pr21872/Homologs/samap_directory/sam1_mouse_w_Supercelltypes.h5ad'
fn2 = '/user/work/pr21872/Homologs/samap_directory/sam2_macaque_w_celltypes.h5ad'

filenames = {'mm':fn1,'mq':fn2}

sam1=SAM()
sam1.load_data(fn1)
print('Loaded mouse')
sam2=SAM()
sam2.load_data(fn2)
print('Loaded macaque')
# #Compute neighbors graphs
# sam1.run()
# sam1.adata.write_h5ad("/user/work/pr21872/Homologs/samap_directory/sam1_mouse.h5ad")
# print('Computed neighborhood mouse graphs')
# sam2.run()
# sam2.adata.write_h5ad("/user/work/pr21872/Homologs/samap_directory/sam2_macaque.h5ad")
# print('Computed neighborhood macque graphs')

sams = {'mm':sam1,'mq':sam2}

sm = SAMAP(
        sams,
        f_maps = '/user/work/pr21872/Homologs/maps/',
    )
print('Created SAMAP')

sm.run()
print('Ran SAMAP')
samap = sm.samap

samap.adata.write_h5ad("/user/work/pr21872/Homologs/samap_directory/mark4_combined_samap.h5ad")

keys = {'mm':'Supertype','mq':'Subclass'}

D, MappingTable = get_mapping_scores(sm, keys, n_top=0)

print(MappingTable.head())
MappingTable.to_csv('/user/work/pr21872/Homologs/samap_directory/CELLMAPPINGS_2.csv')
D.to_csv('/user/work/pr21872/Homologs/samap_directory/CELLMAPPINGSalignments_2.csv')

# Create the scatter plot; this returns an Axes object.
ax = sm.scatter()

# Retrieve the Figure from the Axes.
fig = ax.get_figure()

# Save the figure to a file.
fig.savefig("/user/work/pr21872/Homologs/samap_directory/quality_scatter_2.png", dpi=600, bbox_inches="tight")
# fig.savefig("/user/work/pr21872/Homologs/samap_directory/high_quality_scatter_1.pdf", bbox_inches="tight")

