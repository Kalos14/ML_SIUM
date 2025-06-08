#!/usr/bin/env python
# coding: utf-8

import os
import time
import random
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Impostazioni generali ───────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Directory di output
output_dir = Path("CV_folder")
output_dir.mkdir(exist_ok=True)

# ── Caricamento dati ───────────────────────────────────────────────────────────
# Modifica questi percorsi se servono
dataset_path   = f"/home/{os.environ['USER']}/usa_131_per_size_ranks_False.pkl"
benchmark_path = f"/home/{os.environ['USER']}/SandP benchmark.csv"

stock_data = pd.read_pickle(dataset_path)
stock_data = stock_data[stock_data["size_grp"] == "micro"]
stock_data = stock_data[stock_data["id"] % 6  == 0]
stock_data = stock_data[stock_data["id"] % 5  == 0]

SP_benchmark = pd.read_csv(benchmark_path)
SP_benchmark["caldt"]         = pd.to_datetime(SP_benchmark["caldt"])
SP_benchmark["caldt_period"]  = SP_benchmark["caldt"].dt.to_period("M")

months_list = stock_data["date"].sort_values().unique()
columns_to_drop = ["size_grp", "date", "r_1", "id"]

# ── Definizione rete ────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H = H
        self.W = nn.ParameterList([nn.Parameter(torch.randn(D, D)/100) for _ in range(H)])
        self.V = nn.ParameterList([nn.Parameter(torch.randn(D, D)/100) for _ in range(H)])
    def forward(self, X):
        heads = []
        for h in range(self.H):
            scores = X @ self.W[h] @ X.T / math.sqrt(X.shape[1])
            weights = F.softmax(scores, dim=1) + 1e-8
            A_h     = weights @ X @ self.V[h]
            heads.append(A_h)
        return sum(heads)

class FeedForward(nn.Module):
    def __init__(self, D, dF):
        super().__init__()
        self.fc1     = nn.Linear(D, dF)
        self.fc2     = nn.Linear(dF, D)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, X):
        return self.dropout(self.fc2(F.relu(self.fc1(X))))

class TransformerBlock(nn.Module):
    def __init__(self, D, H, dF):
        super().__init__()
        self.attn  = MultiHeadAttention(D, H)
        self.ffn   = FeedForward(D, dF)
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)
    def forward(self, X):
        X = self.norm1(X + self.attn(X))
        X = self.norm2(X + self.ffn(X))
        return X

class NonlinearPortfolioForward(nn.Module):
    def __init__(self, D, K, H=1, dF=256):
        super().__init__()
        self.blocks     = nn.ModuleList([TransformerBlock(D, H, dF) for _ in range(K)])
        self.lambda_out = nn.Parameter(torch.randn(D, 1)/1000)
    def forward(self, X):
        for block in self.blocks:
            X = block(X)
        w_t = X @ self.lambda_out.squeeze()
        return F.softplus(w_t)

# ── Parametri di training ──────────────────────────────────────────────────────
window         = 60
epochs         = 2
K              = 10
H              = 1
dF             = 256
lr             = 1e-4
ridge_penalties = np.logspace(-4, 0, 5)  # es. [1, 10, 100, 1000]

first_t = window + 1
last_t  = len(months_list) - 1

# ── Loop su ridge penalties ────────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
for ridge in ridge_penalties:
    cum_rets = []

    for t in range(first_t, last_t):
        # Inizializza modello e optimizer *senza* weight_decay
        D = stock_data.shape[1] - len(columns_to_drop)
        model = NonlinearPortfolioForward(D=D, K=K, H=H, dF=dF).to(device)
        optim_ = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

        # Fase di training su rolling window [t-window : t)
        for ep in range(epochs):
            for month in months_list[t-window : t]:
                md     = stock_data[stock_data["date"] == month]
                X_t    = torch.tensor(md.drop(columns=columns_to_drop).values,
                                      dtype=torch.float32, device=device)
                R_next = torch.tensor(md["r_1"].values,
                                      dtype=torch.float32, device=device)
                w_t    = model(X_t)
                loss   = (1 - torch.dot(w_t, R_next)).pow(2) \
                         + ridge * torch.norm(w_t, p=2).pow(2)
                optim_.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim_.step()

        # Fase di test sul periodo t
        with torch.no_grad():
            md      = stock_data[stock_data["date"] == months_list[t]]
            X_t     = torch.tensor(md.drop(columns=columns_to_drop).values,
                                   dtype=torch.float32, device=device)
            R_next  = torch.tensor(md["r_1"].values,
                                   dtype=torch.float32, device=device)
            w_next  = model(X_t)
            ret_t   = (w_next @ R_next).item()
            cum_rets.append(ret_t)
            

    cum_rets = np.cumsum(cum_rets)
    SR = cum_rets.mean()/cum_rets.std() *np.sqrt(12)
    # Plotto la curva cumulata per questa lambda
    plt.plot(months_list[first_t:last_t], cum_rets,
             label=f"ridge={ridge:.0e}, SR ={SR:.0e}")

# Aggiungo S&P 500 per riferimento
dates_period = pd.Series(months_list[first_t:last_t]).astype("datetime64[ns]").tolist()
sp = SP_benchmark[SP_benchmark["caldt_period"]
                  .isin(pd.Series(months_list[first_t:last_t]).astype("period[M]"))]
sp = sp.sort_values("caldt")
sp_cum = np.cumsum(sp["vwretd"].values)
plt.plot(sp["caldt"], sp_cum, "--", color="k", label="S&P 500")

# Formattazione grafico
plt.title("Constrained Portfolio con lr=1e-4, CV su ridge penalties")
plt.xlabel("Time")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=10))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()

# Salvo figura
plt.savefig(output_dir / "cumsum_all_ridge.png")
plt.close()

print("Plot salvato in:", output_dir / "cumsum_all_ridge.png")
