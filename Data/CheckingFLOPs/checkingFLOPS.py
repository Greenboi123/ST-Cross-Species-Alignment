import itertools
import sys
# sys.path.append('/user/work/pr21872/test_aai/Encoder-Decoder')
# from FullModel import MultiTaskTranslationDecoder
sys.path.append('/user/work/pr21872/test_aai')
from HPCEncoderSupertypeEntropy import MultiTaskNumericEncoder, get_sinusoidal_encoding
from ptflops import get_model_complexity_info
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle
import pandas as pd
from joblib import load
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split


# macaque_brain_regions = 141
# decoder = MultiTaskTranslationDecoder(macaque_brain_regions, embed_dim=256, num_heads=4, num_layers=6, dropout=0.1)
# decoder.eval()   # turn off dropout

mouse_brain_regions  = 43  # mouse sequence length
encoder = MultiTaskNumericEncoder(mouse_brain_regions, embed_dim=256, num_heads=4, num_layers=6, dropout=0.1)
encoder.eval()   # turn off dropout

# Dummy mouse data:
batch_size   = 64          # any convenient size
embed_dim    = 256

# with torch.no_grad():
#     macs_all, params_all = get_model_complexity_info(
#         decoder,
#         input_res=(mouse_brain_regions, embed_dim), #(43,256)
#         as_strings=True,
#         print_per_layer_stat=False,
#         verbose=False
#     )

with torch.no_grad():
    macs_all, params_all = get_model_complexity_info(
        encoder,
        input_res=(mouse_brain_regions, ), #(43,256)
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )

print(f"Encoder  MACs : {macs_all}")        # multiply-accumulates
print(f"Encdoer Params: {params_all}")
