#!/usr/bin/env python3
import argparse
import itertools
import os
import pickle
from joblib import load
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/user/work/pr21872')
from test_aai.HPCEncoderSupertypeEntropyRelu import  MultiTaskNumericEncoder, get_sinusoidal_encoding
from sklearn.preprocessing import QuantileTransformer

# Modify the encoder to return the latent
class EncoderFeatureExtractor(MultiTaskNumericEncoder):
    def forward(self, x_hidden):
        x_embed = self.value_embedding(x_hidden.unsqueeze(-1)) + self.pos_enc
        encoder_output = self.transformer_encoder(x_embed)
        return encoder_output

# --- Decoder with the same architecture setup ---
class MultiTaskTranslationDecoder(nn.Module):
    def __init__(self, seq_len=141, embed_dim=256, num_heads=4, num_layers=6, dropout=0.1):
        """
        Note:
            In this design, the decoder does not use causal (masked) self-attention
            because the target is non-sequential. However, it still leverages
            the standard Transformer decoder layer to allow cross attention with
            the pretrained encoders latent representation.
        """
        super().__init__()
        self.seq_len = seq_len
        
        # Learnable query tokens that drive the decoder (fixed at seq_len)
        self.decoder_query = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # Positional encoding for the decoder tokens.
        pos_enc = get_sinusoidal_encoding(seq_len, embed_dim)
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))  # shape: (1, seq_len, embed_dim)
        
        # Not passing any target mask so self-attention is unmasked can just call this once
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True  # use (batch, seq, embed_dim) ordering
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Two heads
        self.classification_head = nn.Linear(embed_dim, 1)
        self.regression_head = nn.Linear(embed_dim, 1)
    
    def forward(self, encoder_latent):
        batch_size = encoder_latent.size(0)
        # Expand the decoder query to match the batch size
        # Add positional encodings
        decoder_input = self.decoder_query.expand(batch_size, -1, -1) + self.pos_enc
        
        # Pass through the Transformer decoder
        # No mask is provided; cross-attention still operates between the decoder input and encoder's latent representation
        decoder_output = self.transformer_decoder(decoder_input, encoder_latent)
        
        # predictions for both heads
        classification_output = self.classification_head(decoder_output).squeeze(-1)
        regression_raw = self.regression_head(decoder_output).squeeze(-1)
        regression_output = F.relu(regression_raw)
        
        return classification_output, regression_output

"""Training"""
def train_decoder(decoder, encoder, train_loader, val_loader, checkpoint_dir,
                  epochs, lr=1e-4, alpha=0.5, device='cuda'):

    # 1) freeze encoder:
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 2) setup decoder optimizer + losses
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = 0.25*steps_per_epoch
    eval_interval = max(1, steps_per_epoch // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            frac = (step - warmup_steps) / (total_steps - warmup_steps)
            return (1 - frac) * 1.0  # decay from 1 → 0, or → (min_lr/initial_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    nan_prcnt = 0.12 
    pos_weight_value = (1-nan_prcnt)/(nan_prcnt)
    pos_weight_tensor = torch.tensor(pos_weight_value).to(device)
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    reg_criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir=checkpoint_dir)
    best_metrics = {}
    i=0

    decoder.to(device)

    for epoch in range(epochs):
        decoder.train()
        for step, batch in enumerate(train_loader):
            mouse_in, maca_in, maca_nan, maca_tgt = batch

            # send to GPU
            mouse_in = mouse_in.to(device)
            maca_in  = maca_in.to(device)
            maca_nan = maca_nan.to(device)
            maca_tgt = maca_tgt.to(device)

            # 3) encode (no grad)
            with torch.no_grad():
                latent = encoder(mouse_in)

            # 4) decode
            cls_out, reg_out = decoder(latent)
            # shapes: (B, 141), (B, 141)

            # 5a) classification loss on macaque_nan 
            loss_cls = cls_criterion(cls_out, maca_nan)

            # 5b) regression on all positions
            loss_reg = reg_criterion(reg_out, maca_tgt)

            # 6) combine & step
            loss = alpha*loss_cls + (1-alpha)*loss_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % eval_interval == 0 or step == steps_per_epoch - 1:
                i+=1
                print(i)
                encoder.eval()
                decoder.eval()
                for phase, loader in [('Train', train_loader), ('Validation', val_loader)]:
                    all_nan_masks, all_targets = [], []
                    all_preds_class, all_preds_reg = [], []

                    with torch.no_grad():
                        for batch in loader:
                            (mouse_in,
                            maca_in,
                            maca_nan,
                            maca_tgt) = batch

                            # to device
                            mouse_in = mouse_in.to(device)
                            maca_nan = maca_nan.to(device)
                            maca_tgt = maca_tgt.to(device)

                            # encode mouse features
                            latent = encoder(mouse_in)   # (B, 43, embed_dim)

                            # decode to predictions
                            pred_class, pred_reg = decoder(latent)  # each (B, 141)

                            # collect for metrics (move to CPU + numpy)
                            all_nan_masks.extend(maca_nan.cpu().numpy().flatten())
                            all_targets.extend(maca_tgt.cpu().numpy().flatten())
                            all_preds_class.extend(pred_class.cpu().numpy().flatten())
                            all_preds_reg.extend(pred_reg.cpu().numpy().flatten())

                    # now compute metrics exactly as before, but using macaque arrays:
                    # classification on all_nan_masks vs all_preds_class
                    cm = np.array(all_nan_masks) >= 0   # all positions are valid here
                    bin_preds = (np.array(all_preds_class)[cm] > 0.5).astype(int)
                    bin_targets = np.array(all_nan_masks)[cm]

                    # regression only where mask == 0 (i.e. observed)
                    rm = np.array(all_nan_masks) == 0
                    reg_preds = np.array(all_preds_reg)[rm]
                    reg_targets = np.array(all_targets)[rm]

                    metrics = {}
                    if len(np.unique(bin_targets)) > 1:
                        metrics.update({
                            'F1':       f1_score(bin_targets, bin_preds),
                            'Precision':precision_score(bin_targets, bin_preds),
                            'Recall':   recall_score(bin_targets, bin_preds),
                            'ROC-AUC':  roc_auc_score(bin_targets, np.array(all_preds_class)[cm]),
                        })
                    if reg_targets.size > 0:
                        metrics.update({
                            'MSE': mean_squared_error(reg_targets, reg_preds),
                            'MAE': mean_absolute_error(reg_targets, reg_preds),
                            'R2':  r2_score(reg_targets, reg_preds),
                        })

                    # log & checkpoint as before
                    for name, val in metrics.items():
                        writer.add_scalar(f'{phase}/{name}', val, epoch*steps_per_epoch + step)
                        if phase == 'Validation' and (
                        name not in best_metrics
                        or (name in ['MSE','MAE'] and val < best_metrics[name])
                        or (name not in ['MSE','MAE'] and val > best_metrics[name])
                        ):
                            best_metrics[name] = val
                            # Unwrap state_dicts if using DataParallel
                            dec_state = (decoder.module.state_dict()
                                        if isinstance(decoder, nn.DataParallel)
                                        else decoder.state_dict())
                            opt_state = optimizer.state_dict()
                            sch_state = scheduler.state_dict()

                            checkpoint = {
                                'decoder_state_dict':       dec_state,
                                'optimizer_state_dict':     opt_state,
                                'scheduler_state_dict':     sch_state,
                                'epoch':                    epoch,
                                'best_metrics':             best_metrics,
                                'args':                     vars(args),
                            }
                            save_dir = os.path.join(checkpoint_dir, 'saved_models')
                            os.makedirs(save_dir, exist_ok=True)
                            torch.save(checkpoint, os.path.join(
                                save_dir, f'checkpoint_dec_{name}_{val:.4f}.pt'
                            ))

                decoder.train()

        scheduler.step()
    writer.close()


"""Initialise the Dataset class to pass to the decoder"""
class DecoderGeneMappingDataset(Dataset):
    def __init__(self,
                 mouse_samples: np.ndarray,        # (160083, 43)
                 macaque_samples: np.ndarray,      # (291060, 141)
                 row_indices: np.ndarray,          # e.g. train_mouse_row_idxs
                 cell_mappings: dict,
                 num_mouse_ct: int = 33,
                 num_macaque_ct: int = 60):
        # build flat list of (mouse_row, macaque_row) pairs
        self.pairs = []
        for m_idx in row_indices:
            gene = m_idx // num_mouse_ct
            m_ct  = m_idx %  num_mouse_ct
            for mq_ct in cell_mappings[m_ct]:
                mq_idx = gene * num_macaque_ct + mq_ct
                self.pairs.append((m_idx, mq_idx))

        self.mouse_samples   = mouse_samples
        self.macaque_samples = macaque_samples

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        m_idx, mq_idx = self.pairs[i]

        # load raw numpy vectors
        raw_mouse   = self.mouse_samples[m_idx]    # shape (43,)
        raw_macaque = self.macaque_samples[mq_idx] # shape (141,)

        # compute NaN masks
        mouse_nan_mask   = torch.from_numpy(np.isnan(raw_mouse).astype(np.float32))
        macaque_nan_mask = torch.from_numpy(np.isnan(raw_macaque).astype(np.float32))

        # build network inputs with NaNs→-1.0
        mouse_input   = torch.from_numpy(raw_mouse.copy()).float()
        macaque_input = torch.from_numpy(raw_macaque.copy()).float()
        mouse_input[mouse_nan_mask.bool()]     = -1.0
        macaque_input[macaque_nan_mask.bool()] = -1.0

        # build regression targets with NaNs→0.0
        # mouse_target   = torch.from_numpy(raw_mouse.copy()).float()
        macaque_target = torch.from_numpy(raw_macaque.copy()).float()
        # mouse_target[mouse_nan_mask.bool()]     = 0.0
        macaque_target[macaque_nan_mask.bool()] = 0.0

        # 5) return everything in a fixed order
        return (
            mouse_input,
            # mouse_nan_mask,
            # mouse_target,
            macaque_input,
            macaque_nan_mask,
            macaque_target,
        )

def main(args):
    """Gather mappings"""   
    cell_indicies_mappings_path = '/user/work/pr21872/test_aai/Encoder-Decoder/cell_mapping_indicies.pkl'
    with open(cell_indicies_mappings_path, 'rb') as f:
        cell_mappings_dict = pickle.load(f)

    macaque_full_celltype_to_idx_path = '/user/work/pr21872/test_aai/Encoder-Decoder/macaque_full_celltype_to_idx.pkl'
    with open(macaque_full_celltype_to_idx_path, 'rb') as f:
        macaque_full_celltype_to_idx = pickle.load(f)

    macaque_celltype_to_idx_path = '/user/work/pr21872/test_aai/Encoder-Decoder/macaque_full_celltype_to_idx.pkl'
    with open(macaque_celltype_to_idx_path, 'rb') as f:
        macaque_celltype_to_idx = pickle.load(f)

    """Gather Datasets"""
    macaque_samples_path = '/user/work/pr21872/test_aai/Encoder-Decoder/CellMapping_Macaque_Samples141Regions.npy'
    mouse_samples_path = '/user/work/pr21872/test_aai/Encoder-Decoder/CellMapping_Mouse_Samples43Regions.npy'
    macaque_mapping_samples = np.load(macaque_samples_path) #(291060, 141) -> 60 cells
    mouse_mapping_samples = np.load(mouse_samples_path) #(160083, 43) -> 33 cells


    """Check prior mouse cells and split mappings by mouse cells"""
    # Paths and constants
    metadata_file = '/user/work/pr21872/SpaceData/bigData/Expressions/metadata/MERFISH-C57BL6J-638850-CCF/20231215/views/cell_metadata_with_parcellation_annotation.csv'
    mouse_cell_to_idx_lookup_path = '/user/work/pr21872/test_aai/Encoder-Decoder/mouse_celltype_to_idx.pkl'

    # Load metadata and filter Iso‑cortex
    metadata = pd.read_csv(metadata_file, usecols=['parcellation_division','supertype','subclass'])
    metadata_iso = metadata[metadata['parcellation_division'] == 'Isocortex']

    # Build supertype -> subclass map and lists of uniques and split the subclasses
    supertype_to_subclass = dict(zip(metadata_iso['supertype'], metadata_iso['subclass']))
    unique_supertypes  = sorted(metadata_iso['supertype'].unique())
    unique_subclasses  = sorted(metadata_iso['subclass'].unique())
    # split the subclass cell types
    mouse_encoder_train_samples, _ = train_test_split(unique_subclasses, test_size=0.2, random_state=42)
    # get all mouse cells used in encoder training
    encoder_subclass_set = set(mouse_encoder_train_samples)

    # Reverse the search
    encoder_supertypes = [
        st for st in unique_supertypes
        if supertype_to_subclass[st] in encoder_subclass_set
    ]

    # Find cell names used in mappings
    with open(mouse_cell_to_idx_lookup_path, 'rb') as f:
        mouse_cell_to_idx = pickle.load(f)
    mouse_mapping_cells = list(mouse_cell_to_idx.keys())

    # Split into “seen by encoder” vs “unseen by encoder”
    cells_in_encoder    = [c for c in mouse_mapping_cells if c in encoder_supertypes]
    print(len(cells_in_encoder)) ## This is 0!
    cells_not_in_encoder= [c for c in mouse_mapping_cells if c not in encoder_supertypes]

    """Before did random split but since so little can just split by eye"""
    decoder_train_cells = ['L2/3 IT CTX Glut_2', 'L2/3 IT ENT Glut_1', 'L2/3 IT ENT Glut_2', 'L2/3 IT ENT Glut_5', 'IT AON-TT-DP Glut_1', 'L5 ET CTX Glut_1', 'L5 ET CTX Glut_2', 'L5 ET CTX Glut_3', 'L5 ET CTX Glut_5', 'L5 ET CTX Glut_6', 'L6b CTX Glut_1', 'L6b CTX Glut_2', 'L6b CTX Glut_3', 'L6b CTX Glut_4', 'L6 CT CTX Glut_2', 'Sncg Gaba_1', 'Sncg Gaba_4', 'Lamp5 Gaba_1', 'Pvalb chandelier Gaba_1', 'Sst Chodl Gaba_2', 'Sst Chodl Gaba_4', 'Astro-TE NN_3', 'OPC NN_1', 'OPC NN_2', 'MOL NN_4', 'VLMC NN_1', 'Endo NN_1']
    decoder_test_cells = ['CLA-EPd-CTX Car3 Glut_1', 'Lamp5 Gaba_4', 'L2/3 IT CTX Glut_4', 'L6b/CT ENT Glut_2', 'Sst Gaba_10','Microglia NN_1']
    # print(len(decoder_train_cells))
    # print(len(decoder_test_cells))
    # get idx mappings
    decoder_train_cells_idx = [mouse_cell_to_idx[cell] for cell in decoder_train_cells]
    decoder_test_cells_idx = [mouse_cell_to_idx[cell] for cell in decoder_test_cells]

    # Mouse test and train data
    # row_index = gene_index * num_celltypes + celltype_index
    # So all_mouse_idxs % num_mouse_celltypes will return each row position
    # Compute which mouse rows go in train/test by celltype
    num_mouse_celltypes = 33
    all_mouse_idxs = np.arange(len(mouse_mapping_samples))
    mouse_ct_idxs = all_mouse_idxs % num_mouse_celltypes
    # Create masks for train/test
    mouse_train_mask = np.isin(mouse_ct_idxs, decoder_train_cells_idx)
    mouse_test_mask  = np.isin(mouse_ct_idxs, decoder_test_cells_idx)
    # Create train/test
    train_mouse_row_idxs = all_mouse_idxs[mouse_train_mask]
    test_mouse_row_idxs  = all_mouse_idxs[mouse_test_mask]

    """Apply the QT transforms to the samples"""
    # -------------- Mouse -------------------
    qt_mouse_encoder_data = load('quantile_transformer_mouse.joblib')
    qt_mouse_data_flat = qt_mouse_encoder_data.transform(mouse_mapping_samples.reshape(-1, 1)).flatten()
    transformed_mouse_mapping_samples = qt_mouse_data_flat.reshape(mouse_mapping_samples.shape) 
    # -------------- Macaque ------------------
    macaque_qt = QuantileTransformer(
    output_distribution='uniform',
    random_state=42,
    n_quantiles=int(1e5),
    subsample=int(1e7) 
    )
    macaque_full_data_path = '/user/work/pr21872/SpaceData/regionsSampleData/float32samples141Regions.npy'
    macaque_full_data = np.load(macaque_full_data_path) # (4108908, 141)
    macaque_full_num_cells = 258 #(15926*258, 141)
    # Don't include the testing macaque cells in the QT transform
    macaque_test_cells_idx = []
    for cell_idx in decoder_test_cells_idx:
        macaque_test_cells_idx.append(cell_mappings_dict[cell_idx])
    macaque_test_cells_idx = list(itertools.chain(*macaque_test_cells_idx))
    # translate between datasets
    macaque_test_cells_names = [cellname for cellname, idx in macaque_celltype_to_idx.items() if idx in macaque_test_cells_idx]
    macaque_idx_in_full_data_for_test_cells = [macaque_full_celltype_to_idx[cellname] for cellname in macaque_test_cells_names]
    # Only include the cell data which are not in the testing of the decoder
    all_full_macaque_idxs = np.arange(len(macaque_full_data))
    macaque_ct_idxs = all_full_macaque_idxs % macaque_full_num_cells
    macaque_decoder_test_mask_on_full_data = np.isin(macaque_ct_idxs, macaque_idx_in_full_data_for_test_cells)
    macaque_samples_able_to_be_in_qt_transform_idx = all_full_macaque_idxs[~macaque_decoder_test_mask_on_full_data]
    macaque_samples_able_to_be_in_qt_transform = macaque_full_data[macaque_samples_able_to_be_in_qt_transform_idx,:]
    # fit QT transform to the full data
    macaque_qt.fit(macaque_samples_able_to_be_in_qt_transform.reshape(-1, 1))
    # apply it to the mapping samples
    transformed_macaque_mapping_samples_flat = macaque_qt.transform(macaque_mapping_samples.reshape(-1, 1)).flatten()
    transformed_macaque_mapping_samples = transformed_macaque_mapping_samples_flat.reshape(macaque_mapping_samples.shape) 

    # Build train & test DataLoaders:
    train_ds = DecoderGeneMappingDataset(
        mouse_samples=transformed_mouse_mapping_samples,
        macaque_samples=transformed_macaque_mapping_samples,
        row_indices=train_mouse_row_idxs,
        cell_mappings=cell_mappings_dict,
        num_mouse_ct=33,
        num_macaque_ct=60
    )
    test_ds = DecoderGeneMappingDataset(
        mouse_samples=transformed_mouse_mapping_samples,
        macaque_samples=transformed_macaque_mapping_samples,
        row_indices=test_mouse_row_idxs,
        cell_mappings=cell_mappings_dict,
        num_mouse_ct=33,
        num_macaque_ct=60
    )
    # Load the datasets
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    # Load and Freeze the pre-trained encoder
    mouse_brain_regions = transformed_mouse_mapping_samples.shape[1]
    encoder_checkpoint_path = '/user/work/pr21872/test_aai/QT_Relu_Tokens/saved_models/checkpoint_best_MSE_0.07286768406629562.pt'
    encoder_checkpoint = torch.load(encoder_checkpoint_path, map_location='cpu')
    # Get the original state_dict
    state_dict = encoder_checkpoint['model_state_dict']
    # Create a new state_dict with 'module.' prefix removed
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    encoder = EncoderFeatureExtractor(mouse_brain_regions, embed_dim=256, num_heads=4, num_layers=6, dropout=0.1)
    encoder.load_state_dict(new_state_dict)

    # Initalise the decoder
    macaque_brain_regions = transformed_macaque_mapping_samples.shape[1]
    decoder = MultiTaskTranslationDecoder(macaque_brain_regions, embed_dim=256, num_heads=4, num_layers=6, dropout=0.1)

    # Load the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    # move onto device
    encoder.to(device)
    decoder.to(device)
    train_decoder(decoder, encoder, train_loader, test_loader, args.checkpoint_dir, args.epochs, args.lr, args.alpha, device=device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a MultiTaskNumericEncoder model.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument('--epochs', type=int, default=7, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--alpha', type=float, default=0.5, help="Loss weighting factor")
    parser.add_argument('--nan_prcnt', type=float, default=0.5, help="Classification weighting factor")
    args = parser.parse_args()
    main(args)