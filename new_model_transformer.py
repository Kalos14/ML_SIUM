import time
from pathlib import Path
from dataclasses import dataclass
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm.auto import tqdm

################################################################################
# 1. Helper utilities
################################################################################

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    return tuple(o.to(device, non_blocking=True) for o in obj)

################################################################################
# 2. Hyper-parameters
################################################################################

@dataclass
class Hyper:
    window: int = 60        # look-back months
    epochs: int = 10        # training epochs per month
    K: int = 10             # number of transformer blocks
    H: int = 1              # attention heads
    dF: int = 256           # feed-forward hidden size
    lr: float = 1e-5        # learning rate
    ridge: float = 10      # ridge penalty
    vol_window: int = 12    # rolling window for vol-managed

################################################################################
# 3. Dataset
################################################################################

class MonthlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feat_cols):
        self.months = [grp for _, grp in df.groupby("date", sort=True)]
        self.feat_cols = feat_cols

    def __len__(self):
        return len(self.months)

    def __getitem__(self, idx):
        g = self.months[idx]
        X = torch.tensor(g[self.feat_cols].values, dtype=torch.float32)
        R = torch.tensor(g["r_1"].values, dtype=torch.float32)
        return X, R

################################################################################
# 4. Model (long-only, no L¹ constraint)
################################################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        assert D % H == 0
        self.H, self.dk = H, D // H
        self.Wqkv = nn.Linear(D, 2 * D, bias=False)
        self.scale = self.dk ** 0.5

    def forward(self, X):
        qk, v = self.Wqkv(X).chunk(2, dim=-1)
        qk = qk.view(X.size(0), self.H, self.dk)
        v  = v.view(X.size(0), self.H, self.dk)
        scores = torch.einsum("nhd,mhd->hnm", qk, qk) / self.scale
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("hnm,mhd->nhd", attn, v)
        return ctx.reshape(X.size(0), -1)

class FeedForward(nn.Module):
    def __init__(self, D, dF):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, dF), nn.ReLU(),
            nn.Linear(dF, D), nn.Dropout(0.1)
        )
    def forward(self, X):
        return self.net(X)

class TransformerBlock(nn.Module):
    def __init__(self, D, H, dF):
        super().__init__()
        self.attn = MultiHeadAttention(D, H)
        self.ffn  = FeedForward(D, dF)
        self.ln1  = nn.LayerNorm(D)
        self.ln2  = nn.LayerNorm(D)

    def forward(self, X):
        X = self.ln1(X + self.attn(X))
        X = self.ln2(X + self.ffn(X))
        return X

class PortfolioTransformer(nn.Module):
    def __init__(self, D, hyper: Hyper):
        super().__init__()
        self.blocks = nn.Sequential(*[
            TransformerBlock(D, hyper.H, hyper.dF)
            for _ in range(hyper.K)
        ])
        self.out = nn.Linear(D, 1, bias=False)

    def forward(self, X):
        logits = self.out(self.blocks(X)).flatten()
        return torch.relu(logits)

################################################################################
# 5. In-place rolling training loop (no scratch refit)
################################################################################

def train_loop(df: pd.DataFrame, hyper: Hyper):
    feat_cols = [c for c in df.columns if c not in {"size_grp","date","r_1","id"}]
    ds = MonthlyDataset(df, feat_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate model & optimizer ONCE
    model = PortfolioTransformer(len(feat_cols), hyper).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyper.lr, weight_decay=1e-5)

    port_ret, dates, losses, roll = [], [], [], deque(maxlen=100)
    step = 0
    uniq_dates = df["date"].unique()

    # loop OOS months with in-place rolling retrain
    for t in tqdm(range(hyper.window, len(ds)-1),
                  desc="OOS months", unit="m"):
        # train on rolling window [t-window, t)
        model.train()
        for _ in range(hyper.epochs):
            for idx in range(t - hyper.window, t):
                X, R = ds[idx]
                X, R = to_device((X, R), device)
                optimizer.zero_grad()
                w = model(X)
                loss = (1 - torch.dot(w, R))**2 + hyper.ridge * w.pow(2).sum()
                loss.backward()
                optimizer.step()
                losses.append((step, loss.item()))
                roll.append(loss.item())
                if step % 500 == 0:
                    print(f"[loss] step {step}, mean100 = {np.mean(roll):.3e}")
                step += 1

        # out-of-sample prediction
        model.eval()
        Xoos, Roos = ds[t]
        Xoos, Roos = to_device((Xoos, Roos), device)
        with torch.no_grad():
            w_oos = model(Xoos)
            port_ret.append(torch.dot(w_oos, Roos).item())
            dates.append(uniq_dates[t+1])

    return dates, port_ret, losses

################################################################################
# 6. Vol-managed helper
################################################################################

def vol_managed(series: pd.Series, window: int):
    return series / series.rolling(window).std().shift(1)

################################################################################
# 7. CLI
################################################################################

def main():
    set_seed()
    df = pd.read_pickle( "our_version_norm.pkl")
    df = df[df["size_grp"] == "micro"].copy()
    df =df[df["id"]% 6 == 0].copy()  

    hyper = Hyper()

    tic = time.time()
    dates, rets, losses = train_loop(df, hyper)
    duration = (time.time() - tic) / 60

    out = Path("project_results_new_model")
    out.mkdir(exist_ok=True)

    # save raw & vol-managed returns
    raw = pd.Series(rets, index=pd.to_datetime(dates), name="Return")
    raw.to_csv(out / "short_constrain_raw.csv")

    man = vol_managed(raw, hyper.vol_window).rename("ManagedReturn")
    man.to_csv(out / "short_constrain_managed.csv")

    # save training loss
    pd.DataFrame(losses, columns=["step","loss"]) \
      .to_csv(out / "training_loss.csv", index=False)

    print(f"Finished in {duration:.1f} min → {len(rets)} OOS months")

if __name__ == "__main__":
    main()
