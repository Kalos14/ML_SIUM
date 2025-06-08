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
output_dir = "positional_encoding_results"
os.makedirs(output_dir, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

dataset_path = f"/home/{os.environ['USER']}/usa_131_per_size_ranks_False.pkl"
benchmark_path = f"/home/{os.environ['USER']}/SandP benchmark.csv"

# --- Load & Prepare Data ---
stock_data = pd.read_pickle(dataset_path)
stock_data = stock_data[stock_data["size_grp"] == "micro"]
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
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):  # [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, D, seq_len, num_layers=4, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(D, D)
        self.pos_enc    = PositionalEncoding(D, dropout, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out      = nn.Linear(D, 1)

    def forward(self, X):  # [batch, seq_len, D]
        X = self.input_proj(X)
        X = self.pos_enc(X)
        X = self.transformer(X)
        return self.fc_out(X[:, -1, :]).squeeze(-1)  # [batch]

class AssetTimeSeriesDataset(Dataset):
    def __init__(self, df, window, features, months):
        self.window = window
        self.features = features
        self.df = df.set_index(['id','date']).sort_index()
        self.samples = []
        for t in range(window, len(months)):
            train_m = months[t-window:t]
            target_m = months[t]
            ids = [set(self.df.loc[pd.IndexSlice[:, m]].index.get_level_values(0)) for m in train_m + [target_m]]
            common = set.intersection(*ids)
            for aid in common:
                seq = [self.df.loc[(aid,m),features].values for m in train_m]
                r = self.df.loc[(aid,target_m),'r_1']
                self.samples.append((np.stack(seq), r))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, r = self.samples[i]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(r, dtype=torch.float32)

# --- Hyperparameters ---
window         = 60
batch_size     = 256
epochs         = 20
ridge_penalty  = 10
lr             = 1e-4
weight_decay   = 1e-5
num_layers     = 4
nhead          = 4
d_ff           = 256
dropout        = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

features = [c for c in stock_data.columns if c not in ['size_grp','date','r_1']]
D = len(features)

# --- DataLoader ---
dataset = AssetTimeSeriesDataset(stock_data, window, features, months_list)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# --- Model & Optimizer ---
model = TimeSeriesTransformer(D, window, num_layers, nhead, d_ff, dropout).to(device)
opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# --- Training Loop with Custom Portfolio Loss ---
for ep in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    for X_seq, R_true in dataloader:
        X_seq, R_true = X_seq.to(device), R_true.to(device)
        preds = model(X_seq)  # [batch]
        # treat preds as portfolio weights before normalization
        exp_p = torch.exp(preds - preds.max())
        w = exp_p / exp_p.sum()
        loss = (1 - torch.dot(w, R_true)).pow(2)
        loss = loss + ridge_penalty * torch.norm(w, p=2).pow(2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total_loss += loss.item() * X_seq.size(0)
    print(f"Epoch {ep}/{epochs} — Loss: {total_loss/len(dataset):.6f}")

# --- Inference & Backtest ---
portfolio_ret = []
dates = []
equally = []
model.eval()
with torch.no_grad():
    for t in range(window, len(months_list)-1):
        seqs, rets = [], []
        m_train = months_list[t-window:t]
        m_test  = months_list[t]
        m_next  = months_list[t+1]
        idx = stock_data.set_index(['id','date'])
        ids = [set(idx.loc[pd.IndexSlice[:,m]].index.get_level_values(0)) for m in m_train]
        common = set.intersection(*ids)
        for aid in common:
            try:
                s = [idx.loc[(aid,m),features].values for m in m_train]
                r = idx.loc[(aid,m_next),'r_1']
            except KeyError: continue
            seqs.append(s); rets.append(r)
        Xe = torch.tensor(np.stack(seqs), dtype=torch.float32, device=device)
        p = model(Xe).cpu()
        exp_p = torch.exp(p - p.max())
        w = exp_p / exp_p.sum()
        portfolio_ret.append((w @ torch.tensor(rets)).item())
        dates.append(m_next)
        equally.append(np.mean(rets))

# --- Save & Plot ---
pd.DataFrame({'Date':dates,'Return':portfolio_ret}).to_csv(os.path.join(output_dir,'portfolio_returns.csv'),index=False)

SP_benchmark['period']=SP_benchmark['caldt'].dt.to_period('M')
per = pd.Series(pd.to_datetime(dates)).dt.to_period('M')
SPr = SP_benchmark[SP_benchmark['period'].isin(per)].sort_values('caldt')

plt.figure(figsize=(10,5))
plt.plot(pd.to_datetime(dates), np.cumsum(portfolio_ret), label='Transformer')
plt.plot(SPr['caldt'], np.cumsum(SPr['vwretd']), '--', label='S&P 500')
plt.plot(pd.to_datetime(dates), np.cumsum(equally), ':', label='Equally')
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(output_dir,'cumulative.png'))
plt.close()
