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
output_dir = "expandedMHA_results"
os.makedirs(output_dir, exist_ok=True)



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
#stock_data = stock_data[stock_data["id"]%6 == 0]

benchmark_path = f"/home/{os.environ['USER']}/SandP benchmark.csv"

SP_benchmark = pd.read_csv(benchmark_path)
SP_benchmark["caldt"] = pd.to_datetime(SP_benchmark["caldt"])

# ## defining transformer structure
# 

# In[16]:
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ExpandedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention across N samples (no sequence dimension).
    This module computes attention weights between all samples in the batch,
    similar to standard self-attention across tokens in a sequence.

    Each head sees all N samples as both queries and keys.
    """

    def __init__(self, d_input, num_heads, init_fn_dict=None, scale_dict=None):
        """
        Parameters:
        - d_input (int): Input feature dimension.
        - num_heads (int): Number of attention heads.
        - init_fn_dict (dict): Optional dict mapping parameter names to init functions.
        - scale_dict (dict): Optional dict mapping parameter names to scale factors.
        """
        super().__init__()

        self.d_input = d_input                   # Original input dimension
        self.num_heads = num_heads               # Number of attention heads
        self.d_proj = d_input * num_heads        # Projected dimension after multi-head concat

        # Linear projections for Q, K, V. These will be reshaped into heads later
        self.W_q = nn.Linear(d_input, self.d_proj)
        self.W_k = nn.Linear(d_input, self.d_proj)
        self.W_v = nn.Linear(d_input, self.d_proj)

        # Final output layer to map multi-head output to scalar prediction
        self.output_layer = nn.Linear(self.d_proj, 1)

        # Optional dictionaries for custom parameter initialization and scaling
        self.init_fn_dict = init_fn_dict or {}
        self.scale_dict = scale_dict or {}

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Applies parameter-specific initialization and scaling.
        Falls back to Xavier initialization and 1.0 scale if unspecified.
        """

        def apply_init(name, param):
            # Use user-specified init function or default to Xavier
            init_fn = self.init_fn_dict.get(name, nn.init.xavier_uniform_)
            init_fn(param)

        def apply_scale(name, param):
            # Multiply weights by a custom scalar (if provided)
            scale = self.scale_dict.get(name, 1.0)
            param.data.mul_(scale)

        # Initialize weights and biases for each projection layer
        for name, layer in zip(['W_q', 'W_k', 'W_v', 'output'],
                               [self.W_q, self.W_k, self.W_v, self.output_layer]):
            apply_init(name + '_weight', layer.weight)  # Initialize weights
            nn.init.zeros_(layer.bias)                  # Zero the bias
            apply_scale(name + '_weight', layer.weight) # Scale the weights

    def forward(self, X, return_attention=False):
        """
        Forward pass of multi-head self-attention.

        Parameters:
        - X: (N, d_input) input tensor
        - return_attention (bool): if True, also return attention weights

        Returns:
        - y: (N, 1) output predictions
        - attn_weights (optional): (N, num_heads, N) attention scores for each head
        """
        N, d = X.shape
        assert d == self.d_input, "Input feature size mismatch"

        # Project input into Q, K, V for all heads: shape (N, h * d_input)
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        # Reshape each into (N, h, d_input) so each head has its own d_input-dim space
        Q = Q.view(N, self.num_heads, self.d_input)
        K = K.view(N, self.num_heads, self.d_input)
        V = V.view(N, self.num_heads, self.d_input)

        # Transpose to (h, N, d_input): each head is a separate batch
        Q = Q.permute(1, 0, 2)
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)

        # Compute raw attention scores with scaled dot product:
        #   (h, N, d) x (h, d, N) -> (h, N, N)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_input ** 0.5)

        # Apply softmax over keys (last dimension): (h, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to values:
        #   (h, N, N) x (h, N, d) -> (h, N, d)
        attn_output = torch.matmul(attn_weights, V)

        # Recombine heads: (h, N, d) -> (N, h, d)
        attn_output = attn_output.permute(1, 0, 2)

        # Flatten across heads: (N, h, d) -> (N, h * d)
        attn_output = attn_output.reshape(N, self.d_proj)

        # Final output projection: (N, h * d) -> (N, 1)
        y = self.output_layer(attn_output)

        # Return predictions and optionally attention weights (transpose to (N, h, N))
        return (y, attn_weights.permute(1, 0, 2)) if return_attention else y


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
        self.attn = ExpandedMultiHeadAttention(D, H)
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
        w_t = X @ self.lambda_out# [N_t, 1]

        return w_t.squeeze()       # [N_t]

# ## Training loop -------

# In[7]:


months_list = stock_data["date"].unique()
columns_to_drop_in_x = ["size_grp", "date", "r_1", "id"]
window = 60
epoch = 20
K = 10
D = stock_data.shape[1] - len(columns_to_drop_in_x)
H = 1
dF = 256
ridge_penalty = 10
lr = 1e-5
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
    

    model_path = os.path.join(output_dir, f"model_weights_t{t}.pt")
    torch.save(model.state_dict(), model_path)
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

csv_path = os.path.join(output_dir, "lele_DF.csv")

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

plt.title(f"Cum Ret: epochs = {epoch}, H = {H}, K = {K}, lr = {lr}, z = {ridge_penalty}")
plt.xlabel("Time")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "myplotSIUM.png"))
plt.close()


