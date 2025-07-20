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

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import itertools

# pre-computed 43×43 Pearson matrix
key_corr = np.corrcoef(attn_avg.T)

seeds   = [2, 10, 42, 50, 100]
k       = 3
labels  = {}            # seed → 1-D array of length 43
silh    = {}            # seed → silhouette score

for seed in seeds:
    # 1. UMAP embedding with this seed
    emb = umap.UMAP(random_state=seed).fit_transform(key_corr)

    # 2. k-means with the *same* seed (optional, you can vary them independently)
    km = KMeans(n_clusters=k, n_init=50, random_state=seed)
    labels[seed] = km.fit_predict(emb)

    # 3. (optional) silhouette per seed
    silh[seed] = silhouette_score(emb, labels[seed])

# 4. pair-wise ARI across all seeds
pairs = list(itertools.combinations(seeds, 2))
aris  = [adjusted_rand_score(labels[s1], labels[s2]) for s1, s2 in pairs]

print(f"Mean pairwise ARI = {np.mean(aris):.2f} ± {np.std(aris):.2f}")
print("Per-seed silhouette:", {s: round(v, 2) for s, v in silh.items()})