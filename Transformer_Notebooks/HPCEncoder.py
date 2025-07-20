#!/usr/bin/env python3
import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

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
    def __init__(self, seq_len, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
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
        class_logits = self.classification_head(encoder_output).squeeze(-1)
        classification_output = torch.sigmoid(class_logits)
        regression_output = self.regression_head(encoder_output).squeeze(-1)
        return classification_output, regression_output

# Training function with metrics logging and checkpointing
def train_model(model, train_loader, val_loader, epochs, checkpoint_dir, lr=1e-4, alpha=0.5, device='cuda'):
    criterion_class = nn.BCELoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    model.to(device)
    writer = SummaryWriter()
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
                                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_best_{metric_name}_{metric_value}.pt')
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

# Prepare data function ensuring at least one hidden position
def prepare_data(data, hidden_fraction=0.3):
    nan_mask = np.isnan(data)
    hidden_mask = np.zeros_like(data)
    num_features = data.shape[1]
    for i in range(len(data)):
        all_indices = np.arange(num_features)
        num_hidden = int(num_features * hidden_fraction)
        hidden_indices = np.random.choice(all_indices, size=num_hidden, replace=False)
        hidden_mask[i, hidden_indices] = 1
    data_original = np.nan_to_num(data, nan=0.0)
    data_hidden = data_original.copy()
    data_hidden[hidden_mask == 1] = 0.0
    data_original = torch.tensor(data_original, dtype=torch.float32)
    data_hidden = torch.tensor(data_hidden, dtype=torch.float32)
    nan_mask_tensor = torch.tensor(nan_mask, dtype=torch.float32)
    training_mask_tensor = torch.tensor(hidden_mask, dtype=torch.float32)
    return data_original, data_hidden, nan_mask_tensor, training_mask_tensor

def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Load data from the provided path (expects a .npy file)
    samples = np.load(args.data_path)
    input_dim = samples.shape[1]
    numSamples = 8460    # total number of samples
    numCellTypes = 104   # number of cell types (rows) per sample
    sample_labels = np.repeat(np.arange(numSamples), numCellTypes)
    sample_indices = np.arange(numSamples)
    train_samples, test_samples = train_test_split(sample_indices, test_size=0.2, random_state=42)
    train_mask = np.isin(sample_labels, train_samples)
    test_mask = np.isin(sample_labels, test_samples)
    del sample_labels
    train_data = samples[train_mask]
    test_data = samples[test_mask]
    del train_mask, test_mask, samples
    np.random.seed(42)
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    train_dataset = MaskedNumericDataset(*prepare_data(train_data))
    test_dataset = MaskedNumericDataset(*prepare_data(test_data))
    del train_data, test_data
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    del train_dataset, test_dataset
    model = MultiTaskNumericEncoder(input_dim, embed_dim=256, num_heads=4, num_layers=6, dropout=0.1)
    train_model(model, train_loader, test_loader, epochs=args.epochs, checkpoint_dir=args.checkpoint_dir, lr=args.lr, alpha=args.alpha)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a MultiTaskNumericEncoder model.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the input .npy data file")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument('--epochs', type=int, default=11, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--alpha', type=float, default=0.5, help="Loss weighting factor")
    args = parser.parse_args()
    main(args)
