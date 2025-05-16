import numpy as np
from scipy.stats import rankdata  # Make sure this is included!
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

df = pd.read_parquet('wrds_data.parquet')


def NonlinearPortfolioTransformer(X_t, parameters):
    # X_t: asset characteristics (N_t x D)
    # parameters: includes all attention, feedforward, and output weights

    Y = X_t
    for k in range(1, K + 1):
        Y = TransformerBlock(Y, parameters[k])

    # Final projection to scalar portfolio weights
    w_t = Y @ parameters['lambda']  # (N_t x D) @ (D x 1) = (N_t x 1)
    return w_t


import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H = H
        self.W = nn.ParameterList([nn.Parameter(torch.randn(D, D) / D) for _ in range(H)])
        self.V = nn.ParameterList([nn.Parameter(torch.randn(D, D) / D) for _ in range(H)])


    def forward(self, X):  # X: [N_t, D]
        heads = []
        for h in range(self.H):
            scores = X @ self.W[h] @ X.T / (X.shape[1] ** 0.5)  # [N_t, N_t]
            weights = F.softmax(scores, dim=1) + 1e-8  # softmax row-wise
            A_h = weights @ X @ self.V[h]  # [N_t, D]
            heads.append(A_h)
        return sum(heads)  # [N_t, D]


class FeedForward(nn.Module):
    def __init__(self, D, dF):
        super().__init__()
        self.fc1 = nn.Linear(D, dF)
        self.fc2 = nn.Linear(dF, D)
        self.dropout = nn.Dropout(p=0.1)
        nn.init.normal_(self.fc1.weight, mean=0, std=1 / dF ** 0.5)
        nn.init.normal_(self.fc2.weight, mean=0, std=1 / D ** 0.5)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, X):  # X: [N_t, D]

        return self.dropout(self.fc2(F.relu(self.fc1(X))))  # [N_t, D]


## old class ##
# class TransformerBlock(nn.Module):
#     def __init__(self, D, H, dF):
#         super().__init__()
#         self.attn = MultiHeadAttention(D, H)
#         self.ffn = FeedForward(D, dF)

#     def forward(self, X):  # X: [N_t, D]
#         X = X + self.attn(X)   # residual after attention
#         X = X + self.ffn(X)    # residual after FFN
#         return X

class TransformerBlock(nn.Module):
    def __init__(self, D, H, dF):
        super().__init__()
        self.attn = MultiHeadAttention(D, H)
        self.ffn = FeedForward(D, dF)
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

    def forward(self, X):  # X: [N_t, D]
        X = self.norm1(X + self.attn(X))  # normalize after attention residual
        X = self.norm2(X + self.ffn(X))  # normalize after FFN residual
        return X


class NonlinearPortfolioForward(nn.Module):
    def __init__(self, D, K, H=1, dF=256):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(D, H, dF) for _ in range(K)])
        self.lambda_out = nn.Parameter(torch.randn(D, 1) / D)


    def forward(self, X):  # X: [N_t, D]
        for block in self.blocks:
            X = block(X)  # propagate through K blocks
        w_t = X @ self.lambda_out  # [N_t, 1]

        return w_t.squeeze()  # [N_t]

def drop_col(df, threshold_max_na):
    """
    Drops inplace the columns of a dataframe who have more than "threshold" percent of Nan
    """
    return df.loc[:,df.isna().mean(axis = 0)<threshold_max_na]


def drop_permno_with_too_many_nans(df):
    """
    Identifies permno values of rows with more than 50% NaNs,
    then removes all rows corresponding to these permnos.
    Moreover it deletes stocks with only 15 observations
    """
    # Step 1: Find rows where more than 1/3% of the columns have NaN values
    rows_with_many_nans = df[df.isna().mean(axis=1) > 1/3]

    # Step 2: Get unique permno values for those rows
    permno_to_drop = rows_with_many_nans["permno"].unique()

    # Step 3: Remove all rows with these permno values
    df = df[~df["permno"].isin(permno_to_drop)].copy().reset_index()

    #sic value cannot be Nan
    #df_filtered = df.dropna(subset=['sic'])

    return df  # Returning the filtered df and dropped permno values for reference

df_clean = df[df["size_grp"] != "nano"] #delete nano stock as in the paper
df_clean = drop_col(df_clean, 0.25)
df_clean = drop_permno_with_too_many_nans(df_clean)
df_clean = df_clean.drop(columns=['size_grp', "sic", "ff49"]) #useelss for X_t
df_clean = df_clean.dropna(subset=["ret_exc_lead1m"])

exclude_cols = ['index', 'eom', 'permno']
ranked = df_clean.drop(columns=exclude_cols).rank(axis=0, method='average', pct=True) - 0.5

# Step 2: Replace any remaining NaNs with 0
ranked = ranked.fillna(0)

# Combine back with the excluded columns
df_normalized = pd.concat([df_clean[exclude_cols], ranked], axis=1)

months_list = df_clean["eom"].unique()
columns_to_drop_in_x = ["index", "eom", "ret_exc_lead1m", "permno"]
window = 60
epoch = 2
K = 4
D = df_normalized.shape[1] - len(columns_to_drop_in_x)
H = 40
dF = 32
t = 100  # --> 61st month in the months_list

lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

model = NonlinearPortfolioForward(D=D, K=K, H=H, dF=dF).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

losses = []
returns = []
z= 1/1000
for e in range(epoch):
    for month in months_list[t - window:t]:  ## this loop iterates til t-1
        X_t = df_normalized[df_normalized["eom"] == month].drop(columns=columns_to_drop_in_x)
        R_t_plus_one = torch.tensor(df_clean[df_clean["eom"] == month]["ret_exc_lead1m"].values, dtype=torch.float32,
                                    device=device)
        X_t_tensor = torch.tensor(X_t.values, dtype=torch.float32, device=device)  # Shape: [N_t, D]

        w_t = model(X_t_tensor)  # Shape: [N_t]

        loss = (1 - torch.dot(w_t, R_t_plus_one)).pow(2) + z*(model.lambda_out**2).sum()

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


import matplotlib.pyplot as plt

# Optional: downsample
step = 10
plt.figure(figsize=(10, 6))
plt.plot(losses[::step], label='Losses', color='red')
plt.plot(returns[::step], label='Returns', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Losses and Returns per Iteration')
plt.legend()
plt.grid(True)

# Show the plot instead of saving
plt.show()

# Print minimum
min_loss = min(losses)
min_index = losses.index(min_loss)
print(f"Smallest loss: {min_loss:.6f}")
print(f"Corresponding return: {returns[min_index]:.6f}")


#### prediction at t #####
month_t = months_list[t]
X_t = df_normalized[df_normalized["eom"] == month_t].drop(columns=columns_to_drop_in_x)
R_t_plus_one = torch.tensor(df_clean[df_clean["eom"] == month_t]["ret_exc_lead1m"].values, dtype=torch.float32,
                            device=device)
X_t_tensor = torch.tensor(X_t.values, dtype=torch.float32, device=device)  # Shape: [N_t, D]

w_t = model(X_t_tensor)  # Shape: [N_t]
## does it actually work?
print(f"return at t:{torch.dot(w_t, R_t_plus_one).item()}")