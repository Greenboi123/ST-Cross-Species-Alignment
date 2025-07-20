import ast
import torch
import numpy as np
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

RS = 42
# ────────────────────────────────
# 1️⃣  LOAD DATA
# ────────────────────────────────
dictsave_path = '/user/work/pr21872/test_aai/attention_heads/avg_attn_weights_one_head.pt'
region_file   = '/user/work/pr21872/test_aai/Encoder-Decoder/Mouse_Regions_List.txt'

attn_dict   = torch.load(dictsave_path, map_location='cpu')
layer_keys  = sorted(attn_dict.keys(), key=lambda k: int(k.replace('layer', '')))
all_layers  = [attn_dict[k] for k in layer_keys]
stacked     = torch.stack(all_layers, dim=0)          # → [L, H, S, S]

layer_idx, head_idx = 0, 0                            # ← tweak if you like
attn_avg   = stacked[layer_idx, head_idx]             # [seq, seq]

with open(region_file, 'r') as f:
    brain_region_labels = ast.literal_eval(f.read())

# ────────────────────────────────
# 2️⃣  PREP FIGURE ─ LEFT: ATTENTION
# ────────────────────────────────
attn_matrix = attn_avg.numpy()                        # heat-map data

# ────────────────────────────────
# 3️⃣  PREP FIGURE ─ RIGHT: UMAP + K-MEANS
# ────────────────────────────────
key_corr = np.corrcoef(attn_avg.T)                    # correlations across keys
embedding = umap.UMAP(random_state=RS).fit_transform(key_corr)

k       = 3                                           # number of clusters
labels  = KMeans(n_clusters=k, n_init=50,
                 random_state=RS).fit_predict(embedding)

# ────────────────────────────────
# 4️⃣  PLOT SIDE-BY-SIDE
# ────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

# ── 4️⃣  PLOT SIDE-BY-SIDE ──────────────────────────────────────────
plt.rcParams.update({                 # ← global defaults
    "font.size": 12,                 # tick labels & text()
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
})

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))  # a bit wider

# --- left panel: attention heat-map --------------------------------
im = ax1.imshow(attn_matrix, aspect='equal', cmap='viridis')
cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel('Attention weight', fontsize=12)

ax1.set_xlabel('Key position')
ax1.set_ylabel('Query position')
ax1.set_title('Layer 1, Head 1 – Average Attention', pad=12)

# --- right panel: UMAP scatter -------------------------------------
k = 3
palette = sns.color_palette("Set2", k)
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1],
                hue=labels, palette=palette,
                s=25, alpha=1, edgecolor="k", linewidth=0.2, ax=ax2)

for i, region in enumerate(brain_region_labels):
    ax2.text(embedding[i, 0],
             embedding[i, 1],
             rf"{region}$_{{{i}}}$",
             fontsize=10, ha='center', va='center')

ax2.set_title(f'UMAP of key correlations coloured by K-Means (k = {k})', pad=12)
ax2.set_xlabel('UMAP-1')
ax2.set_ylabel('UMAP-2')

ax2.legend(title='Cluster', loc='best',
           markerscale=1.4, frameon=False)

from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=RS)
pca_embedding = pca.fit_transform(key_corr)  
k = 3                                                 # keep identical to UMAP run
pca_labels = KMeans(n_clusters=k, n_init=50,
                    random_state=RS).fit_predict(pca_embedding)

# ------------------------------------------------------------------
# 3)  Plot – reuse the same palette & labelling style
#     (create an extra axis to the right of your existing ax2)
# ------------------------------------------------------------------
palette = sns.color_palette("Set2", k)

# --- 1) make a copy of the cluster labels for plotting ----------------
vis_labels = pca_labels.copy()

sns.scatterplot(
    x=pca_embedding[:, 0], y=pca_embedding[:, 1],
    hue=vis_labels, palette=palette,
    s=25, alpha=1, edgecolor="k", linewidth=0.2, ax=ax3
)

for i, region in enumerate(brain_region_labels):
    ax3.text(pca_embedding[i, 0],
             pca_embedding[i, 1],
             rf"{region}$_{{{i}}}$",
             fontsize=10, ha='center', va='center')

ax3.set_title(f'PCA of key correlations coloured by K-Means (k = {k})', pad=12)
ax3.set_xlabel('PC-1 ({:.1f}% var)'.format(pca.explained_variance_ratio_[0]*100))
ax3.set_ylabel('PC-2 ({:.1f}% var)'.format(pca.explained_variance_ratio_[1]*100))

ax3.legend(title='Cluster', loc='best',
           markerscale=1.4, frameon=False)

# Optional: tighten axis ranges to match UMAP aesthetics
ax3.set_aspect('equal', adjustable='datalim')

plt.tight_layout()
plt.savefig('/user/work/pr21872/test_aai/attention_heads/attn_and_umap.png',
            dpi=600, bbox_inches='tight')

