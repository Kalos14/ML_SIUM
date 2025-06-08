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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Configuration ---
output_dir = "project_results"
os.makedirs(output_dir, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Paths ---
dataset_path   = f"/home/{os.environ['USER']}/our_version_norm.pkl"
benchmark_path = f"/home/{os.environ['USER']}/SandP benchmark.csv"

# --- Load Data ---
stock_data = pd.read_pickle(dataset_path)
stock_data = stock_data[stock_data["size_grp"] == "micro"]
stock_data["date"] = pd.to_datetime(stock_data["date"])
months_list = sorted(stock_data["date"].unique())

SP_benchmark = pd.read_csv(benchmark_path)
SP_benchmark["caldt"] = pd.to_datetime(SP_benchmark["caldt"])

# --- Model Definition ---
class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H = H
        self.W = nn.ParameterList([nn.Parameter(torch.randn(D, D)/100) for _ in range(H)])
        self.V = nn.ParameterList([nn.Parameter(torch.randn(D, D)/100) for _ in range(H)])

    def forward(self, X):  # [N_t, D]
        heads = []
        for h in range(self.H):
            scores = X @ self.W[h] @ X.T / (X.shape[1] ** 0.5)
            weights = F.softmax(scores, dim=1) + 1e-8
            A_h = weights @ X @ self.V[h]
            heads.append(A_h)
        return sum(heads)

class FeedForward(nn.Module):
    def __init__(self, D, dF):
        super().__init__()
        self.fc1 = nn.Linear(D, dF)
        self.fc2 = nn.Linear(dF, D)
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

    def forward(self, X):  # [N_t, D]
        for block in self.blocks:
            X = block(X)
        w_t = X @ self.lambda_out  # [N_t, 1]
        return w_t.squeeze()       # [N_t]

# --- Hyperparameters ---
window         = 60
epochs         = 30
K              = 10
D              = stock_data.shape[1] - len(["size_grp", "date", "r_1", "id"])
H              = 1
dF             = 256
ridge_penalty  = 10
lr             = 1e-5
device         = "cuda" if torch.cuda.is_available() else "cpu"
columns_to_drop = ["size_grp", "date", "r_1", "id"]

# --- Walk-Forward Training & Inference ---
equally_weighted = []
portfolio_ret    = []
dates_to_save    = []

first_t = window + 1
last_T  = len(months_list) - 1
for t in range(first_t, last_T):
    # Initialize model fresh for each fold
    model = NonlinearPortfolioForward(D=D, K=K, H=H, dF=dF).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Train on the past `window` months
    for e in range(epochs):
        for month in months_list[t-window:t]:
            month_data = stock_data[stock_data["date"] == month]
            X_t = month_data.drop(columns=columns_to_drop)
            R_next = torch.tensor(month_data["r_1"].values, dtype=torch.float32, device=device)
            X_t_tensor = torch.tensor(X_t.values, dtype=torch.float32, device=device)

            w_t = model(X_t_tensor)
            loss = (1 - torch.dot(w_t, R_next)).pow(2)
            loss = loss + ridge_penalty * torch.norm(w_t, p=2).pow(2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # Test on month t+1
    test_month = months_list[t]
    next_month = months_list[t+1]
    test_data = stock_data[stock_data["date"] == test_month]

    X_test = test_data.drop(columns=columns_to_drop)
    R_next = torch.tensor(test_data["r_1"].values, dtype=torch.float32, device=device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32, device=device)

    w_t = model(X_test_tensor)
    port_ret = (w_t @ R_next).item()

    portfolio_ret.append(port_ret)
    dates_to_save.append(next_month)
    equally_weighted.append(R_next.mean().item())

    # Save model weights for this fold
    torch.save(model.state_dict(), os.path.join(output_dir, f"model_weights_t{t}.pt"))

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
eq_cum = np.cumsum(np.asarray(equally_weighted)[:len(SP_cum)])

plt.figure(figsize=(10, 5))
plt.plot(pd.to_datetime(dates_to_save)[:len(SP_cum)], port_cum, label="Transformer Portfolio")
plt.plot(SP_ret["caldt"], SP_cum, linestyle="--", label="S&P 500")
plt.plot(pd.to_datetime(dates_to_save)[:len(SP_cum)], eq_cum, linestyle=":", label="Equally Weighted")

plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.title(f"Cumulative Returns: epochs={epochs}, H={H}, K={K}, lr={lr}, ridge={ridge_penalty}")
plt.xlabel("Time")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cumulative_returns.png"))
plt.close()
