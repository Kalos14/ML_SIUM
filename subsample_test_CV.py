#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from pathlib import Path
from scipy.stats import rankdata  # Make sure this is included!
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import os
start_time = time.time()


### SUPERCOMPUTER ###
output_dir = Path("project_results_constrained")
output_dir.mkdir(exist_ok=True)

print("you are running the code about weights constraint")

def set_seed(seed=42):
    random.seed(seed)  # Python
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs


set_seed(42)

# In[2]:

dataset_path = f"/home/{os.environ['USER']}/usa_131_per_size_ranks_False.pkl"
stock_data = pd.read_pickle(dataset_path)

stock_data = stock_data[stock_data["size_grp"] == "micro"]
## Creating the subsampled dataset
stock_data = stock_data[stock_data["id"]%6==0]
stock_data = stock_data[stock_data["id"]%5==0]

benchmark_path = f"/home/{os.environ['USER']}/SandP benchmark.csv"

SP_benchmark = pd.read_csv(benchmark_path)
SP_benchmark["caldt"] = pd.to_datetime(SP_benchmark["caldt"])


# ## defining transformer structure
# 

# In[16]:

def factor_ret(signals, next_period_ret):
    """"
    Inputs are signals_t and r_t+1, hence this needs to be in a loop for each month
    """
    return (signals.values*next_period_ret.values).sum()/signals.values.sum()

class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H = H
        self.W = nn.ParameterList([nn.Parameter(torch.randn(D, D)/100) for _ in range(H)])
        self.V = nn.ParameterList([nn.Parameter(torch.randn(D, D)/100) for _ in range(H)])

    def forward(self, X):  # X: [N_t, D]
        heads = []
        for h in range(self.H):

            scores = X @ self.W[h] @ X.T/(X.shape[1] ** 0.5)          # [N_t, N_t]
            weights = F.softmax(scores, dim=1) + 1e-8      # softmax row-wise
            A_h = weights @ X @ self.V[h]           # [N_t, D]
            heads.append(A_h)
        return sum(heads)                           # [N_t, D]

class FeedForward(nn.Module):
    def __init__(self, D, dF):
        super().__init__()
        self.fc1 = nn.Linear(D, dF)
        self.fc2 = nn.Linear(dF, D)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X):  # X: [N_t, D]

        return self.dropout(self.fc2(F.relu(self.fc1(X))))  # [N_t, D]

    
class TransformerBlock(nn.Module):
    def __init__(self, D, H, dF):
        super().__init__()
        self.attn = MultiHeadAttention(D, H)
        self.ffn = FeedForward(D, dF)
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, X):  # X: [N_t, D]
        X = self.norm1(X + self.attn(X))  # normalize after attention residual
        X = self.norm2(X + self.ffn(X))   # normalize after FFN residual
        return X

    
class NonlinearPortfolioForward(nn.Module):
    def __init__(self, D, K, H=1, dF=256):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(D, H, dF) for _ in range(K)])
        self.lambda_out = nn.Parameter(torch.randn(D, 1)/1000)  # final projection

    def forward(self, X):  # X: [N_t, D]
        for block in self.blocks:
            X = block(X)  # propagate through K blocks
        w_t = X @ self.lambda_out.squeeze()# [N_t]
        w_t = torch.relu(w_t)  # Ensure non-negativity
        if w_t.sum() > 0:
            w_t = w_t / w_t.sum()
        return w_t    # [N_t]

# ## Training loop -------

# In[7]:


months_list = stock_data["date"].unique()
columns_to_drop_in_x = ["size_grp", "date", "r_1", "id"]


X_non_num = stock_data[columns_to_drop_in_x]


import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Prepara tutto ciò che serve una volta sola
SP_benchmark["caldt"] = pd.to_datetime(SP_benchmark["caldt"])
SP_benchmark["caldt_period"] = SP_benchmark["caldt"].dt.to_period("M")

output_dir = "cartella_output"
os.makedirs(output_dir, exist_ok=True)

window = 60
epoch = 4
K = 10
D = stock_data.shape[1] - len(columns_to_drop_in_x)
H = 1
dF = 256

ridge_penalties = np.logspace(0, 3, 4)
lrs = np.logspace(-7, -5, 3)
device = "cuda" if torch.cuda.is_available() else "cpu"

first_t = 61
last_T = len(months_list) - 2

for ridge_penalty in ridge_penalties:
    # Prepara la figura per questo specifico ridge_penalty
    plt.figure(figsize=(12, 6))

    # Per ogni combinazione di learning rate
    for lr in lrs:
        portfolio_ret = []
        dates_to_save = []

        # Loop temporale “train on past / predict next month”
        for t in range(first_t, last_T):
            model = NonlinearPortfolioForward(D=D, K=K, H=H, dF=dF).to(device)
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=1e-4  
            )

            # FASE DI TRAIN
            for e in range(epoch):
                for month in months_list[t - window : t]:
                    month_data = stock_data[stock_data["date"] == month]
                    X_t = month_data.drop(columns=columns_to_drop_in_x)
                    R_t_plus_one = torch.tensor(
                        month_data["r_1"].values,
                        dtype=torch.float32,
                        device=device
                    )
                    X_t_tensor = torch.tensor(X_t.values, dtype=torch.float32, device=device)
                    w_t = model(X_t_tensor)

                    loss = (1 - torch.dot(w_t, R_t_plus_one)).pow(2) \
                           + ridge_penalty * torch.norm(w_t, p=2).pow(2)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            # FASE DI INFERENZA OOS per t+1
            next_month_data = stock_data[stock_data["date"] == months_list[t]]
            X_next = next_month_data.drop(columns=columns_to_drop_in_x)
            R_next = torch.tensor(
                next_month_data["r_1"].values,
                dtype=torch.float32,
                device=device
            )
            X_next_tensor = torch.tensor(X_next.values, dtype=torch.float32, device=device)
            w_next = model(X_next_tensor)

            predicted_ret = (w_next @ R_next).item()
            portfolio_ret.append(predicted_ret)
            dates_to_save.append(months_list[t + 1])

        # Calcolo dello Sharpe Ratio usando la serie di ritorni predetti
        ret_array = np.array(portfolio_ret)
        mean = ret_array.mean()
        std = ret_array.std(ddof=1)
        sharpe_ratio = np.sqrt(12) * mean / std

        # Preparo il benchmark corrispondente alle date_to_save
        dates_period = pd.Series(dates_to_save).dt.to_period("M")
        SP_ret = SP_benchmark[SP_benchmark["caldt_period"].isin(dates_period)]
        SP_ret = SP_ret.sort_values("caldt")
        SP_cum_return = np.cumsum(SP_ret["vwretd"].values)

        # Plot del benchmark (una volta sola, ad esempio solo per lr il primo)
        if lr == lrs[0]:
            plt.plot(SP_ret["caldt"].values, SP_cum_return, linestyle="--", label="S&P 500")

        # Plot del portafoglio con questo lr
        portfolio_cum_return = np.cumsum(ret_array)
        plt.plot(
            dates_to_save,
            portfolio_cum_return,
            label=f"LR={lr}, Sharpe={sharpe_ratio:.2f}"
        )

    # Titolo e formattazione asssi
    plt.title(
        f"Constrained Portfolio: epochs={epoch}, H={H}, K={K}, "
        f"ridge={ridge_penalty:.3e}"
    )
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Salvo il plot per questo ridge_penalty
    plt.savefig(os.path.join(output_dir, f"constrained_ridge_{ridge_penalty:.3e}.png"))
    plt.close()








end_time = time.time()
print(f"\nExecution time: {(end_time - start_time)/60:.2f} minutes")




