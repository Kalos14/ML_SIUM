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
import datetime as dt


# In[2]:


stock_data = pd.read_pickle("./usa_131_per_size_ranks_False.pkl")
stock_data


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

# class MultiHeadAttention(nn.Module):
#     def __init__(self, D, H):
#         super().__init__()
#         self.D = D
#         self.H = H
#         self.d_proj = D * H

#         # Q, K, V linear projections (shared dimensions)
#         self.W_q = nn.Linear(D, self.d_proj)
#         self.W_k = nn.Linear(D, self.d_proj)
#         self.W_v = nn.Linear(D, self.d_proj)

#         # optional final projection (can keep or skip)
#         self.out_proj = nn.Linear(self.d_proj, D)

#     def forward(self, X):  # X: [N_t, D]
#         N = X.size(0)

#         Q = self.W_q(X).view(N, self.H, self.D)  # [N, H, D]
#         K = self.W_k(X).view(N, self.H, self.D)
#         V = self.W_v(X).view(N, self.H, self.D)

#         attn_scores = torch.einsum('nhd,mhd->nhm', Q, K) / (self.D ** 0.5)  # [N, H, N]
#         attn_weights = torch.softmax(attn_scores, dim=-1)  # attention across assets

#         context = torch.einsum('nhm,mhd->nhd', attn_weights, V)  # [N, H, D]
#         context = context.reshape(N, -1)  # [N, H*D]

#         return self.out_proj(context)  # [N, D]

    

    
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
        w_t = X @ self.lambda_out# [N_t, 1]

        return w_t.squeeze()/w_t.sum()       # [N_t]


# ## fitting one datapoint
# 

# In[4]:


X_t = stock_data[stock_data["date"] == pd.Timestamp("2020-12-31")].drop(columns = ["r_1", "id", "date", "size_grp"])
R_t_plus_one = stock_data[stock_data["date"] == pd.Timestamp("2020-12-31")][["r_1"]]
print("X_t:\n")
print(X_t)
print("R_t+1")
print(R_t_plus_one)


# ## Training loop -------

# In[7]:


months_list = stock_data["date"].unique()
columns_to_drop_in_x = ["size_grp", "date", "r_1", "id"]
window = 60
epoch = 1
K = 2
D = stock_data.shape[1] - len(columns_to_drop_in_x)
H = 2
dF = 64
t = 150  # --> 61st month in the months_list

lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

model = NonlinearPortfolioForward(D=D, K=K, H=H, dF=dF).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

losses = []
returns = []

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

    loss = (1 - torch.dot(w_t, R_t_plus_one)).pow(2)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for name, param in model.named_parameters():
        if torch.isnan(param.grad).any():
            print(f"NaN detected in gradient of {name}")
            break

    portfolio_return = torch.dot(w_t, R_t_plus_one).item()
    losses.append(loss.item())
    returns.append(portfolio_return)

    optimizer.step()

    print(f"  month {month}  loss={loss.item():.6f} return={w_t @ R_t_plus_one}")


# ### now it has been trained from t-60 to t-1, time to give him informations at t and get the weights w_t.
# To see if it fucking worked than do w_t @ R_t_plus_1

# In[12]:


month_data = stock_data[stock_data["date"] == months_list[t]]

X_t = month_data.drop(columns=columns_to_drop_in_x)

R_t_plus_one = torch.tensor(
    month_data["r_1"].values,
    dtype=torch.float32,
    device=device
)

X_t_tensor = torch.tensor(X_t.values, dtype=torch.float32, device=device)  # Shape: [N_t, D]
w_t = model(X_t_tensor)  # Shape: [N_t]
w_t@R_t_plus_one


# # NOW LET'S DO THIS FOR BUNCH OF t 

# In[23]:


portfolio_ret = []
for t in range(200, len(months_list)):
    model = NonlinearPortfolioForward(D=D, K=K, H=H, dF=dF).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    losses = []
    returns = []
    print(months_list[t])

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

        loss = (1 - torch.dot(w_t, R_t_plus_one)).pow(2)

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
    print(predicted)



# In[ ]:





# In[22]:


import matplotlib.pyplot as plt

cumulative_return = np.cumsum(np.asarray(portfolio_ret))

plt.plot(cumulative_return)
plt.title("Cumulative Portfolio Return")
plt.xlabel("Time Step")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.show()


# In[24]:


jupyter nbconvert --to script 2605test.ipynb

