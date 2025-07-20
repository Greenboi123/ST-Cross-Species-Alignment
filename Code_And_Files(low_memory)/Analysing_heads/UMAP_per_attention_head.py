import ast
import torch
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns


with open('/user/work/pr21872/test_aai/Encoder-Decoder/Mouse_Regions_List.txt','r') as myFile:
    data = myFile.read()
brain_region_labels = ast.literal_eval(str(data))
dictsave_path = '/user/work/pr21872/test_aai/attention_heads/avg_attn_weights_one_head.pt'
attn_dict = torch.load(dictsave_path, map_location='cpu')

# stack into [layers, heads, seq, seq]:
layer_keys = sorted(attn_dict.keys(), key=lambda k: int(k.replace('layer','')))
all_layers  = [attn_dict[k] for k in layer_keys]
stacked     = torch.stack(all_layers, dim=0)  # → [L, H, S, S]

num_layers, num_heads, seq_len, _ = stacked.shape

# for layer_idx in range(num_layers):
#     # Create a 2×num_heads grid
#     fig, axes = plt.subplots(
#         2, num_heads,
#         figsize=(4 * num_heads, 8),
#         constrained_layout=True
#     )

#     for head_idx in range(num_heads):
#         # --- Row 0: Square UMAP scatter ---
#         ax_umap = axes[0, head_idx]
#         attn_avg = stacked[layer_idx, head_idx].cpu().numpy()
#         key_corr = np.corrcoef(attn_avg.T)

#         embedding = umap.UMAP(random_state=42).fit_transform(key_corr)

#         # scatter
#         ax_umap.scatter(embedding[:, 0], embedding[:, 1], s=10)

#         # compute a shared limit
#         xmin, xmax = embedding[:, 0].min(), embedding[:, 0].max()
#         ymin, ymax = embedding[:, 1].min(), embedding[:, 1].max()
#         lim_min = min(xmin, ymin)
#         lim_max = max(xmax, ymax)

#         ax_umap.set_xlim(lim_min, lim_max)
#         ax_umap.set_ylim(lim_min, lim_max)
#         ax_umap.set_aspect('equal', 'box')    # square axes

#         # offset labels slightly
#         x_offset = (lim_max - lim_min) * 0.02
#         for i, region in enumerate(brain_region_labels):
#             label = rf"{region}$_{{{i}}}$"
#             ax_umap.text(
#                 embedding[i, 0] + x_offset,
#                 embedding[i, 1],
#                 label,
#                 fontsize=6,
#                 ha='left',
#                 va='center'
#             )

#         ax_umap.set_title(f'Layer {layer_idx+1} • Head {head_idx+1}\n(UMAP)')
#         ax_umap.set_xlabel('UMAP1')
#         ax_umap.set_ylabel('UMAP2')

#         # --- Row 1: Attention heatmap ---
#         ax_heat = axes[1, head_idx]
#         sns.heatmap(
#             attn_avg,
#             cmap='viridis',
#             square=True,
#             cbar=True,
#             ax=ax_heat,
#             xticklabels=10,
#             yticklabels=10
#         )
#         ax_heat.set_title(f'Layer {layer_idx+1} • Head {head_idx+1}\n(Attention)')
#         ax_heat.set_xlabel('Key position')
#         ax_heat.set_ylabel('Query position')

#     # Super‐title and save
#     fig.suptitle(f'Layer {layer_idx+1}: UMAP vs. Attention Heatmaps', y=1.02)
#     out_path = f'/user/work/pr21872/test_aai/attention_heads/one_attention_overview_layer{layer_idx+1}.png'
#     fig.savefig(out_path, dpi=600)
#     plt.close(fig)
#     print(f"Saved {out_path}")

layer_idx = 0    # zero-based
head_idx  = 0    # zero-based
# fig_path = f'/user/work/pr21872/test_aai/attention_heads/umap_L{layer_idx}_H{head_idx}.png'

# for layer_idx in range(6):
#     for head_idx in range(4):
fig_path = f'/user/work/pr21872/test_aai/attention_heads/One_head_umap_L{layer_idx+1}_H{head_idx+1}.png'

attn_avg = stacked[layer_idx, head_idx]

# Compute correlation across key columns
key_corr = np.corrcoef(attn_avg.T)  #corrcoef treats each row as variable, so need to transpose to get keys as variables.

# UMAP embedding
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(key_corr)

# # Plot UMAP embedding
# plt.figure(figsize=(8,6))
# sns.scatterplot(x=embedding[:,0], y=embedding[:,1], alpha=1, s=8)

# # Optionally label points with brain regions if you have labels
# for i, region in enumerate(brain_region_labels):
#     # region name, then subscripted index in math mode
#     label = rf"{region}$_{{{i}}}$"
#     plt.text(
#         embedding[i, 0],
#         embedding[i, 1],
#         label,
#         fontsize=8,
#         ha='center',
#         va='center'
#     )

# plt.title('UMAP of Key-position Correlations (Attention)')
# plt.xlabel('UMAP1')
# plt.ylabel('UMAP2')
# plt.tight_layout()
# plt.savefig(fig_path, dpi=600)

# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# kmeans = KMeans(n_clusters=3, n_init=50).fit(embedding)
# labels  = kmeans.labels_

# sil   = silhouette_score(embedding, labels)
# ch    = calinski_harabasz_score(embedding, labels)
# dbi   = davies_bouldin_score(embedding, labels)

# print(f"silhouette={sil:.2f},  CH={ch:.1f},  DBI={dbi:.2f}")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------------------------------------------
# 1️⃣  K-means on the chosen k  (here k=3) and coloured UMAP scatter
# ------------------------------------------------------------------
k = 3
kmeans = KMeans(n_clusters=k, n_init=50, random_state=42).fit(embedding)
labels = kmeans.labels_

plt.figure(figsize=(8, 6))
palette = sns.color_palette("Set2", k)
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1],
                hue=labels, palette=palette, s=11, alpha=1, edgecolor="k", linewidth=0)

for i, region in enumerate(brain_region_labels):
    # region name, then subscripted index in math mode
    label = rf"{region}$_{{{i}}}$"
    plt.text(
        embedding[i, 0],
        embedding[i, 1],
        label,
        fontsize=8,
        ha='center',
        va='center'
    )

plt.title(f'UMAP of key correlations coloured by K-Means (k={k})')
plt.xlabel('UMAP-1'); plt.ylabel('UMAP-2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig('/user/work/pr21872/test_aai/attention_heads/umap_kmeans_k3.png', dpi=600)



# ------------------------------------------------------------------
# 2️⃣  “Elbow” curve: inertia (within-cluster SSE) vs. k
# ------------------------------------------------------------------
max_k = 5
inertias = []
silhouettes = []

for kk in range(2, max_k + 1):
    km = KMeans(n_clusters=kk, n_init=50, random_state=42).fit(embedding)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(embedding, km.labels_))

fig, ax1 = plt.subplots(figsize=(7, 4))

ax1.plot(range(2, max_k + 1), inertias, marker='o')
ax1.set_xlabel('Number of clusters k')
ax1.set_ylabel('Inertia (SSE)')
ax1.set_title('Elbow plot for K-Means')
fig.savefig('/user/work/pr21872/test_aai/attention_heads/kmeans_elbow.png', dpi=300)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

outlier_idx = 19                 # zero-based index to remove
mask        = np.ones(embedding.shape[0], dtype=bool)
mask[outlier_idx] = False

embed_clean   = embedding[mask]          # shape [n_keys-1, 2]

# ----- K-means & metrics on the cleaned array -----
k     = 3
km    = KMeans(n_clusters=k, n_init=50, random_state=42).fit(embed_clean)
labels_clean = km.labels_

sil   = silhouette_score(embed_clean, labels_clean)
dbi   = davies_bouldin_score(embed_clean, labels_clean)

print(f"silhouette={sil:.2f},  Davies–Bouldin={dbi:.2f}")