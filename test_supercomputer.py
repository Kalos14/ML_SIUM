#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.stats import rankdata  # Make sure this is included!
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import datetime as dt
import os
### SUPERCOMPUTER ###
output_dir = "project_results_mega"
os.makedirs(output_dir, exist_ok=True)



def set_seed(seed=42):
    random.seed(seed)  # Python
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs


set_seed(42)

# In[2]:

dataset_path = f"/home/{os.environ['USER']}/our_version_norm.pkl"
stock_data = pd.read_pickle(dataset_path)

stock_data = stock_data[stock_data["size_grp"] == "mega"]

benchmark_path = f"/home/{os.environ['USER']}/SandP benchmark.csv"

SP_benchmark = pd.read_csv(benchmark_path)
SP_benchmark["caldt"] = pd.to_datetime(SP_benchmark["caldt"])

# ## defining transformer structure
# 

# In[16]:


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
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        for block in self.blocks:
            X = block(X)
        w_t = X @ self.lambda_out.squeeze()
        return w_t
# ## Training loop -------

# In[7]:


months_list = stock_data["date"].unique()
columns_to_drop_in_x = ["size_grp", "date", "r_1", "id"]
window = 60
epoch = 12
K = 10
D = stock_data.shape[1] - len(columns_to_drop_in_x)
H = 1
dF = 256
ridge_penalty = 10
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"


equally_weighted = []
portfolio_ret = []
dates_to_save = []
first_t = 61
last_T =  len(months_list)-2 #first_t + 10
for t in range(first_t, last_T):
    model = NonlinearPortfolioForward(D=D, K=K, H=H, dF=dF).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    losses = []
    returns = []
    #print(months_list[t])
    for e in range(epoch):
        for month in months_list[t - window:t]:  # this loop iterates until t-1
            month_data = stock_data[stock_data["date"] == month]

            X_t = month_data.drop(columns=columns_to_drop_in_x)

            R_t_plus_one = torch.tensor(
                month_data["r_1"].values,
                dtype=torch.float32,
                device=device
            )

            X_t_tensor = torch.tensor(X_t.values, dtype=torch.float32, device=device)  # Shape: [N_t, D]
            w_t = model(X_t_tensor)  # Shape: [N_t]

            loss = (1 - torch.dot(w_t, R_t_plus_one)).pow(2) + ridge_penalty * torch.norm(w_t, p=2).pow(2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            portfolio_return = torch.dot(w_t, R_t_plus_one).item()

            losses.append(loss.item())
            returns.append(portfolio_return)

            optimizer.step()

        #print(f"  month {month}  loss={loss.item():.6f} return={w_t @ R_t_plus_one}")
    month_data = stock_data[stock_data["date"] == months_list[t]]

    X_t = month_data.drop(columns=columns_to_drop_in_x)

    R_t_plus_one = torch.tensor(
        month_data["r_1"].values,
        dtype=torch.float32,
        device=device
    )

    X_t_tensor = torch.tensor(X_t.values, dtype=torch.float32, device=device)  # Shape: [N_t, D]
    w_t = model(X_t_tensor)  # Shape: [N_t]

    predicted = (w_t @ R_t_plus_one).item()
    portfolio_ret.append(predicted)
    dates_to_save.append(months_list[t+1])
    equally_weighted.append(R_t_plus_one.mean().item())
    


    #print(predicted, R_t_plus_one.mean().item() )



# In[ ]:


# Step 1: Create a dictionary
data = {
    "Date": dates_to_save,
    "Return": portfolio_ret
}

# Step 2: Convert to DataFrame
lele = pd.DataFrame(data)

# Step 3: Export to CSV

csv_path = os.path.join(output_dir, "lele_DF_mega.csv")

lele.to_csv(csv_path, index=False)
# In[22]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SP_benchmark["caldt"] = pd.to_datetime(SP_benchmark["caldt"])
dates_to_save = pd.to_datetime(dates_to_save)

# Create monthly periods
SP_benchmark["caldt_period"] = SP_benchmark["caldt"].dt.to_period("M")
dates_period = pd.Series(dates_to_save).dt.to_period("M")

# Filter SP rows to only those with matching year/month
SP_ret = SP_benchmark[SP_benchmark["caldt_period"].isin(dates_period)]

# Sort and align
SP_ret = SP_ret.sort_values("caldt")
SP_cum_return = np.cumsum(SP_ret["vwretd"].values)

# Use the same date order for plotting
aligned_dates = SP_ret["caldt"].values
portfolio_cum_return = np.cumsum(np.asarray(portfolio_ret)[:len(aligned_dates)])

equally_weighted = np.asarray(equally_weighted)[:len(aligned_dates)]
equally_cum_return = np.cumsum(equally_weighted)


plt.figure(figsize=(10, 5))
plt.plot(dates_to_save, portfolio_cum_return, label="Transformer Portfolio")
plt.plot(dates_to_save, SP_cum_return, label="S&P 500", linestyle="--")
plt.plot(dates_to_save, equally_cum_return, label="Equally Weighted", linestyle=":")

plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=10))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.title(f"Cum Ret: epochs = {20}, H = {H}, K = {K}, lr = {lr}, z = {ridge_penalty}")
plt.xlabel("Time")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "myplotSIUM_mega.png"))
plt.close()




