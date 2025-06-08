#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Configuration & Paths ---
output_dir = "project_results"
os.makedirs(output_dir, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Paths (adjust as needed)
dataset_path = f"/home/{os.environ['USER']}/our_version_norm.pkl"
benchmark_path = f"/home/{os.environ['USER']}/SandP benchmark.csv"

# --- Load Data ---
stock_data = pd.read_pickle(dataset_path)
# filter as before
stock_data = stock_data[stock_data["size_grp"] == "micro"]
# parse dates
stock_data["date"] = pd.to_datetime(stock_data["date"])
months_list = sorted(stock_data["date"].unique())

SP_benchmark = pd.read_csv(benchmark_path)
SP_benchmark["caldt"] = pd.to_datetime(SP_benchmark["caldt"])

# --- Model Components ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, D, seq_len, num_layers=2, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(D, D)
        self.pos_encoder = PositionalEncoding(D, dropout, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(D, 1)

    def forward(self, X):  # X: [batch, seq_len, D]
        X = self.input_proj(X)
        X = self.pos_encoder(X)
        X = self.transformer(X)
        out = self.fc_out(X[:, -1, :])  # last timestep
        return out.squeeze(-1)

class AssetTimeSeriesDataset(Dataset):
    def __init__(self, df, window, features, months):
        self.window = window
        self.features = features
        self.months = months
        self.df = df.set_index(['id', 'date']).sort_index()
        self.samples = []
        for t in range(window, len(months)):
            train_months = months[t-window:t]
            target_month = months[t]
            # assets present in all training + target
            ids_sets = [set(self.df.loc[pd.IndexSlice[:, m]].index.get_level_values(0)) for m in train_months + [target_month]]
            common_ids = set.intersection(*ids_sets)
            for asset in common_ids:
                seq = [self.df.loc[(asset, m), self.features].values for m in train_months]
                target = self.df.loc[(asset, target_month), 'r_1']
                self.samples.append((np.stack(seq), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# --- Hyperparameters ---
window       = 60
batch_size   = 256
epochs       = 20
lr           = 1e-4
weight_decay = 1e-5
num_layers   = 4
nhead        = 4
d_ff         = 256
dropout      = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Prepare features list
cols_to_drop = ['size_grp', 'date', 'r_1']
features = [c for c in stock_data.columns if c not in cols_to_drop]
D = len(features)

# --- Dataset & DataLoader ---
dataset = AssetTimeSeriesDataset(stock_data, window, features, months_list)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# --- Model, Optimizer, Loss ---
model = TimeSeriesTransformer(D=D, seq_len=window,
                              num_layers=num_layers, nhead=nhead,
                              dim_feedforward=d_ff, dropout=dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()



# --- Training Loop ---
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    for X_seq, y in dataloader:
        X_seq, y = X_seq.to(device), y.to(device)
        preds = model(X_seq)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X_seq.size(0)
    avg_loss = total_loss / len(dataset)
    print(f'Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f}')

# --- Inference & Portfolio Construction ---
portfolio_ret = []
dates_to_save = []
equally_weighted = []

model.eval()
with torch.no_grad():
    for t in range(window, len(months_list)-1):
        train_months = months_list[t-window:t]
        eval_month    = months_list[t]
        next_month    = months_list[t+1]

        # build eval sequences
        df_idx = stock_data.set_index(['id', 'date'])
        ids_sets = [set(df_idx.loc[pd.IndexSlice[:, m]].index.get_level_values(0)) for m in train_months]
        common_ids = set.intersection(*ids_sets)
        seqs, rets = [], []
        for asset in common_ids:
            try:
                seq = [df_idx.loc[(asset, m), features].values for m in train_months]
                ret = df_idx.loc[(asset, next_month), 'r_1']
            except KeyError:
                continue
            seqs.append(seq)
            rets.append(ret)
        if not seqs:
            continue

        X_eval = torch.tensor(np.stack(seqs), dtype=torch.float32, device=device)
        preds = model(X_eval).cpu().numpy()
        rets = np.array(rets)

        # softmax weights\        
        exp_preds = np.exp(preds - np.max(preds))
        weights = exp_preds / exp_preds.sum()
        port_ret = (weights * rets).sum()

        portfolio_ret.append(port_ret)
        dates_to_save.append(next_month)
        equally_weighted.append(rets.mean())

# --- Export Results ---
results_df = pd.DataFrame({
    'Date': dates_to_save,
    'Return': portfolio_ret
})
results_df.to_csv(os.path.join(output_dir, 'portfolio_returns.csv'), index=False)

# --- Plot Cumulative Returns ---
SP_benchmark["caldt_period"] = SP_benchmark["caldt"].dt.to_period("M")
dates_period = pd.Series(pd.to_datetime(dates_to_save)).dt.to_period("M")
SP_ret = SP_benchmark[SP_benchmark["caldt_period"].isin(dates_period)].sort_values("caldt")
SP_cum = np.cumsum(SP_ret["vwretd"].values)
port_cum = np.cumsum(np.asarray(portfolio_ret)[:len(SP_cum)])
eq
ew_eq_cum = np.cumsum(np.asarray(equally_weighted)[:len(SP_cum)])

plt.figure(figsize=(10,5))
plt.plot(pd.to_datetime(dates_to_save)[:len(SP_cum)], port_cum, label="Transformer Portfolio")
plt.plot(SP_ret["caldt"], SP_cum, linestyle="--", label="S&P 500")
plt.plot(pd.to_datetime(dates_to_save)[:len(SP_cum)], new_eq_cum, linestyle=":", label="Equally Weighted")

plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.title("Cumulative Returns: Transformer vs Benchmarks")
plt.xlabel("Time")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cumulative_returns.png"))
plt.close()
