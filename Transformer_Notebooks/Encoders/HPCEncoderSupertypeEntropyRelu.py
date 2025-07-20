#!/usr/bin/env python3
import argparse
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import subprocess
from sklearn.preprocessing import QuantileTransformer


# Sinusoidal Positional Encoding
def get_sinusoidal_encoding(seq_len, embed_dim):
    pe = torch.zeros(seq_len, embed_dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Improved Multi-task Transformer Encoder 
class MultiTaskNumericEncoder(nn.Module):
    def __init__(self, seq_len, embed_dim=128, num_heads=4, num_layers=6, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.value_embedding = nn.Linear(1, embed_dim)
        pos_enc = get_sinusoidal_encoding(seq_len, embed_dim)
        self.register_buffer("pos_enc", pos_enc.unsqueeze(0))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classification_head = nn.Linear(embed_dim, 1)
        self.regression_head = nn.Linear(embed_dim, 1)

    def forward(self, x_hidden):
        x_embed = self.value_embedding(x_hidden.unsqueeze(-1)) + self.pos_enc
        encoder_output = self.transformer_encoder(x_embed)
        classification_output = self.classification_head(encoder_output).squeeze(-1)
        # classification_output = torch.sigmoid(class_logits) # BCEWithLogitsLoss sorts this
        regression_raw = self.regression_head(encoder_output).squeeze(-1)
        regression_output = F.relu(regression_raw) # no non-negetives
        return classification_output, regression_output

# Training function with metrics logging and checkpointing
def train_model(model, train_loader, val_loader, epochs, checkpoint_dir, lr=1e-4, alpha=0.5, nan_prcnt=0.6, device='cuda'):
    pos_weight_value = (1-nan_prcnt)/(nan_prcnt)
    pos_weight_tensor = torch.tensor(pos_weight_value).to(device)
    criterion_class = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    criterion_reg = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6)
    writer = SummaryWriter(log_dir=checkpoint_dir)
    best_metrics = {}
    steps_per_epoch = len(train_loader)
    eval_interval = max(1, steps_per_epoch // 10)
    i = 0

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            x_filled = batch['x_hidden'].to(device)
            nan_mask = batch['nan_mask'].to(device)
            training_mask = batch['training_mask'].to(device)
            x_original = batch['x_original'].to(device)

            classification_output, regression_output = model(x_filled)
            
            class_loss = criterion_class(classification_output, nan_mask)
            reg_loss = criterion_reg(regression_output * training_mask, x_original * training_mask)
            total_loss = alpha * class_loss + (1 - alpha) * reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % eval_interval == 0 or step == steps_per_epoch - 1:
                print(f"Eval Number: {i}")
                i += 1
                model.eval()
                for phase, loader in [('Train', train_loader), ('Validation', val_loader)]:
                    all_nan_masks, all_train_masks = [], []
                    all_targets, all_preds_class, all_preds_reg = [], [], []

                    with torch.no_grad():
                        for batch in loader:
                            x_filled = batch['x_hidden'].to(device)
                            nan_mask = batch['nan_mask'].to(device)
                            training_mask = batch['training_mask'].to(device)
                            x_original = batch['x_original'].to(device)

                            val_class_out, val_reg_out = model(x_filled)

                            all_nan_masks.extend(nan_mask.cpu().numpy().flatten())
                            all_train_masks.extend(training_mask.cpu().numpy().flatten())
                            all_targets.extend(x_original.cpu().numpy().flatten())
                            all_preds_class.extend(val_class_out.cpu().numpy().flatten())
                            all_preds_reg.extend(val_reg_out.cpu().numpy().flatten())

                    # Classification metrics: Evaluate original missingness prediction (nan_mask)
                    class_mask_indices = np.array(all_nan_masks) >= 0
                    bin_preds = (np.array(all_preds_class)[class_mask_indices] > 0.5).astype(int)
                    bin_targets = np.array(all_nan_masks)[class_mask_indices]
                    # Regression metrics: Evaluate only on hidden-for-training positions
                    reg_mask_indices = np.array(all_train_masks) == 1
                    reg_preds = np.array(all_preds_reg)[reg_mask_indices]
                    reg_targets = np.array(all_targets)[reg_mask_indices]
                    metrics = {}

                    if len(np.unique(bin_targets)) > 1:
                        metrics.update({
                            'F1': f1_score(bin_targets, bin_preds),
                            'Precision': precision_score(bin_targets, bin_preds),
                            'Recall': recall_score(bin_targets, bin_preds),
                            'ROC-AUC': roc_auc_score(bin_targets, np.array(all_preds_class)[class_mask_indices])
                        })
                    if reg_targets.size > 0:
                        metrics.update({
                            'MSE': mean_squared_error(reg_targets, reg_preds),
                            'MAE': mean_absolute_error(reg_targets, reg_preds),
                            'R2': r2_score(reg_targets, reg_preds)
                        })

                    for metric_name, metric_value in metrics.items():
                        writer.add_scalar(f'{phase}/{metric_name}', metric_value, epoch * steps_per_epoch + step)
                        if phase == 'Validation':
                            if (metric_name not in best_metrics) or \
                               ((metric_name in ['MSE', 'MAE']) and (metric_value < best_metrics[metric_name])) or \
                               ((metric_name not in ['MSE', 'MAE']) and (metric_value > best_metrics[metric_name])):
                                best_metrics[metric_name] = metric_value
                                checkpoint = {
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                }
                                save_dir = os.path.join(checkpoint_dir, 'saved_models')
                                os.makedirs(save_dir, exist_ok=True)
                                checkpoint_path = os.path.join(save_dir, f'checkpoint_best_{metric_name}_{metric_value}.pt')
                                torch.save(checkpoint, checkpoint_path)
                    model.train()
        scheduler.step()
    writer.close()

class MaskedNumericDataset(Dataset):
    def __init__(self, data_original, data_hidden, nan_mask_tensor, training_mask_tensor):
        self.data_original = data_original
        self.data_hidden = data_hidden
        self.nan_mask = nan_mask_tensor
        self.training_mask = training_mask_tensor

    def __len__(self):
        return self.data_original.shape[0]

    def __getitem__(self, idx):
        return {
            'x_original': self.data_original[idx],
            'x_hidden': self.data_hidden[idx],
            'nan_mask': self.nan_mask[idx],
            'training_mask': self.training_mask[idx]
        }

def prepare_data(data, hidden_fraction=0.15):
    # Create a mask indicating which positions were originally NaN
    nan_mask = np.isnan(data).astype(np.float32)
    # Initialize a hidden mask (all zeros) with the same shape as the data
    hidden_mask = np.zeros_like(data, dtype=np.float32)
    
    # Determine the total number of features per sample
    num_features = data.shape[1]
    num_hidden = int(num_features * hidden_fraction)
    
    for i in range(len(data)):
        # Use all indices (both NaNs and values)
        all_indices = np.arange(num_features)
        
        # Randomly select indices from all available indices
        hidden_indices = np.random.choice(all_indices, size=num_hidden, replace=False)
        hidden_mask[i, hidden_indices] = 1
    
    # Replace NaN values with -1.0 for classification help
    data_original = np.nan_to_num(data, nan=-1.0)

    # set hidden positions to 0.0 with 10% time random value and 10% time unchanged
    data_hidden = data_original.copy()

    # Get all hidden positions
    current_vals = data_hidden[hidden_mask == 1]  # shape: (total_hidden_positions,)

    # Generate random numbers for each hidden element
    r = np.random.rand(current_vals.size)  # one random value per hidden position
    rand_vals = np.random.uniform(0.0, 1.0, size=current_vals.size)

    # Apply the rules:
    # - if r < 0.1: use a random value
    # - if 0.1 <= r < 0.2: leave unchanged (keep current_vals)
    # - if r >= 0.2: set to 2.0
    new_vals = np.where(r < 0.1, rand_vals, 
                        np.where(r >= 0.2, 2.0, current_vals))

    # Update the data_hidden array at the hidden positions
    data_hidden[hidden_mask == 1] = new_vals

    # For regression NaN = 0.0 
    data_original_eval = data_original.copy()
    data_original_eval[data_original_eval == -1.0] = 0.0

    data_original_eval = torch.tensor(data_original_eval, dtype=torch.float32)
    data_hidden = torch.tensor(data_hidden, dtype=torch.float32)
    nan_mask_tensor = torch.tensor(nan_mask, dtype=torch.float32)
    training_mask_tensor = torch.tensor(hidden_mask, dtype=torch.float32)
    return data_original_eval, data_hidden, nan_mask_tensor, training_mask_tensor

def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Load data from the provided path (expects a .npy file)
    samples = np.load(args.data_path)
    # Permuted so each row is # row_index = gene_index * num_celltypes + celltype_index
    print('loaded')
    brain_regions = samples.shape[1]
    num_genes = 8460

    # Need to split by higher level subclass
    metadata_file = '/user/work/pr21872/SpaceData/bigData/Expressions/metadata/MERFISH-C57BL6J-638850-CCF/20231215/views/cell_metadata_with_parcellation_annotation.csv'
    metadata = pd.read_csv(metadata_file, usecols=['parcellation_division','supertype','subclass'])
    # Filter to only cells in the Isocortex division
    metadata_isocortex = metadata[metadata['parcellation_division'] == 'Isocortex'].copy()
    # Create a mapping from supertype to subclass using the metadata
    supertype_to_subclass = dict(zip(metadata_isocortex['supertype'], metadata_isocortex['subclass']))
    unique_supertypes = sorted(metadata_isocortex['supertype'].unique()) 
    unique_subclasses = sorted(metadata_isocortex['subclass'].unique()) 
    # Now create the new list with subclass values corresponding to each supertype
    repeated_subclass_list = [supertype_to_subclass[st] for st in unique_supertypes]
    categorical = pd.Categorical(repeated_subclass_list)
    cell_integer_categories = categorical.codes
    # Creates a repeating pattern to match the permuted data and relate it to higher level subclass
    cell_repeated_categories = np.tile(cell_integer_categories, num_genes)
    subclass_labels = np.arange(len(unique_subclasses))

    #split the subclass cell types
    train_samples, test_samples = train_test_split(subclass_labels, test_size=0.2, random_state=42)
    train_mask = np.isin(cell_repeated_categories, train_samples)
    test_mask = np.isin(cell_repeated_categories, test_samples)
    del cell_repeated_categories
    train_data = samples[train_mask]
    test_data = samples[test_mask]
    del train_mask, test_mask, samples
    # Apply QT on the train data (no data leakage high entropy)
    qt = QuantileTransformer(
    output_distribution='uniform',
    random_state=42,
    n_quantiles=int(1e5),
    subsample=int(10e6) 
    )
    transformed_train_flat = qt.fit_transform(train_data.reshape(-1, 1)).flatten()
    transformed_train_data = transformed_train_flat.reshape(train_data.shape) 
    # Apply same transform to test data
    transformed_test_flat = qt.transform(test_data.reshape(-1, 1)).flatten()
    transformed_test_data = transformed_test_flat.reshape(test_data.shape) 

    # Shuffle the data
    np.random.seed(42)
    np.random.shuffle(transformed_train_data)
    np.random.shuffle(transformed_test_data)
    # Ready data for training
    train_dataset = MaskedNumericDataset(*prepare_data(transformed_train_data))
    test_dataset = MaskedNumericDataset(*prepare_data(transformed_test_data))
    del train_data, test_data
    ## One node training: ###
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    del train_dataset, test_dataset
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MultiTaskNumericEncoder(brain_regions, embed_dim=256, num_heads=4, num_layers=6, dropout=0.1)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    train_model(model, train_loader, test_loader, epochs=args.epochs, checkpoint_dir=args.checkpoint_dir, lr=args.lr, alpha=args.alpha, nan_prcnt=args.nan_prcnt, device=device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a MultiTaskNumericEncoder model.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input .npy data file")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument('--epochs', type=int, default=11, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--alpha', type=float, default=0.5, help="Loss weighting factor")
    parser.add_argument('--nan_prcnt', type=float, default=0.5, help="Classification weighting factor")
    args = parser.parse_args()
    main(args)


    # local_rank = setup_distributed()
    # device = torch.device(f'cuda:{local_rank}')
    # # Create and move the model to the correct device.
    # model = MultiTaskNumericEncoder(input_dim, embed_dim=256, num_heads=4, num_layers=6, dropout=0.1)
    # model.to(device)
    # print('heh')
    # # Wrap with DistributedDataParallel.
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # print('lol')
    # # Set up distributed samplers for your datasets.
    # train_sampler = DistributedSampler(train_dataset)
    # test_sampler = DistributedSampler(test_dataset, shuffle=False)
    # train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
    # test_loader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler)
    # print('bah')
    # # Train your model.
    # train_model(model, train_loader, test_loader, epochs=args.epochs, 
    #             checkpoint_dir=args.checkpoint_dir, lr=args.lr, alpha=args.alpha, device=device)

# import os
# import subprocess
# import torch
# import torch.distributed as dist

# def setup_distributed():
#     # Set MASTER_ADDR and MASTER_PORT if not already set.
#     if 'MASTER_ADDR' not in os.environ:
#         master_addr = subprocess.check_output(
#             "scontrol show hostname $SLURM_NODELIST | head -n 1", shell=True
#         ).decode().strip()
#         os.environ['MASTER_ADDR'] = master_addr
#         os.environ['MASTER_PORT'] = '29500'  # choose an appropriate port

#     # Print the master information for debugging.
#     print("MASTER_ADDR:", os.environ.get("MASTER_ADDR"), flush=True)
#     print("MASTER_PORT:", os.environ.get("MASTER_PORT"), flush=True)

#     # Get and print SLURM-related environment variables.
#     world_size = int(os.environ.get('SLURM_NTASKS', 1))
#     rank = int(os.environ.get('SLURM_PROCID', 0))
#     local_rank = int(os.environ.get('SLURM_LOCALID', 0))
#     print("SLURM_NTASKS (world_size):", world_size, flush=True)
#     print("SLURM_PROCID (global rank):", rank, flush=True)
#     print("SLURM_LOCALID (local rank):", local_rank, flush=True)

#     # Initialize the process group.
#     dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
#     print("Process group initialized.", flush=True)

#     # Set the device based on local rank.
#     torch.cuda.set_device(local_rank)
#     current_device = torch.cuda.current_device()
#     print("Current CUDA device (should equal local_rank):", current_device, flush=True)
#     print("Device name:", torch.cuda.get_device_name(current_device), flush=True)

#     return local_rank
