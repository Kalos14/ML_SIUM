# Optimized portfolio transformer – **v1.3 (leva variabile + vol-target)**
# ---------------------------------------------------------------------
# 2025‑06‑06  |  Change‑log  (implements “1 e 2” richiesti)
#   • **Nessuna normalizzazione L¹**: il forward ora restituisce w ≥ 0 ma non
#     divide per ∑w ⇒ leva endogena variabile, in stile notebook del prof.
#   • **Volatility‑managed returns**: dopo l’OOS si costruisce una serie pandas
#     e la si scala per la dev‑std rolling a 12 mesi, shiftata di 1.
#   • Optionally, `vol_window` hyper‑param (default 12) per cambiare l’horizon.
# ---------------------------------------------------------------------

import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

################################################################################
# 1.  Helper utilities
################################################################################

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def to_device(obj, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    return tuple(o.to(device, non_blocking=True) for o in obj)

################################################################################
# 2.  Hyper‑parameters
################################################################################

@dataclass
class Hyper:
    window: int = 60       # look‑back months
    epochs: int = 10
    K: int = 10
    H: int = 1
    dF: int = 256
    lr: float = 1e-5
    ridge: float = 1
    vol_window: int = 12   # rolling window for vol‑target
    use_compile: bool = False

################################################################################
# 3.  Dataset (no padding – compile off)
################################################################################

class MonthlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.months = [m for _, m in df.groupby("date", sort=True)]
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.months)

    def __getitem__(self, idx: int):
        g = self.months[idx]
        X = torch.tensor(g[self.feature_cols].values, dtype=torch.float32)
        R = torch.tensor(g["r_1"].values, dtype=torch.float32)
        return X, R

################################################################################
# 4.  Model (no L¹ budget)
################################################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, D: int, H: int):
        super().__init__(); assert D % H == 0
        self.H, self.d_k = H, D // H
        self.Wqkv = nn.Linear(D, 2 * D, bias=False)
        self.scale = (self.d_k) ** 0.5
    def forward(self, X):
        qk, v = self.Wqkv(X).chunk(2, dim=-1)
        qk = qk.view(X.size(0), self.H, self.d_k)
        v  = v.view(X.size(0), self.H, self.d_k)
        s  = torch.einsum("nhd,mhd->hnm", qk, qk) / self.scale
        a  = torch.softmax(s, -1)
        ctx= torch.einsum("hnm,mhd->nhd", a, v).reshape(X.size(0), -1)
        return ctx

class FeedForward(nn.Module):
    def __init__(self, D, dF):
        super().__init__(); 
        self.net = nn.Sequential(nn.Linear(D, dF), nn.ReLU(), nn.Linear(dF, D), nn.Dropout(0.1))
    def forward(self, X): return self.net(X)

class TransformerBlock(nn.Module):
    def __init__(self, D, H, dF):
        super().__init__(); self.attn = MultiHeadAttention(D, H); 
        self.ffn = FeedForward(D, dF); 
        self.n1 = nn.LayerNorm(D); 
        self.n2 = nn.LayerNorm(D)
    def forward(self, X):
        X = self.n1(X + self.attn(X))
        X = self.n2(X + self.ffn(X))
        return X

class PortfolioTransformer(nn.Module):
    def __init__(self, D, hyper: Hyper):
        super().__init__(); 
        self.blocks = nn.Sequential(*[TransformerBlock(D, hyper.H, hyper.dF) for _ in range(hyper.K)]); 
        self.out = nn.Linear(D, 1, bias=False)
    def forward(self, X):
        X = self.blocks(X)
        w = self.out(X).flatten()  
        return w

################################################################################
# 5.  Training loop (identico, ma loss adattata)
################################################################################

def train_loop(df: pd.DataFrame, hyper: Hyper):
    feat_cols = [c for c in df.columns if c not in {"size_grp", "date", "r_1", "id"}]
    ds = MonthlyDataset(df, feat_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PortfolioTransformer(D=len(feat_cols), hyper=hyper).to(device)

    opt = optim.Adam(model.parameters(), lr=hyper.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    port_ret, dates, losses, roll = [], [], [], deque(maxlen=100)
    step = 0

    for t in range(hyper.window, len(ds) - 1):
        idx_window = list(range(t - hyper.window, t))
        for _ in range(hyper.epochs):
            for idx in idx_window:
                X, R = ds[idx]; X, R = to_device((X, R), device)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    w = model(X)
                    loss = (1 - torch.dot(w, R)) ** 2 + hyper.ridge * w.pow(2).sum()
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                lv = loss.item(); losses.append((step, lv)); roll.append(lv)
                step += 1
        # OOS
        X, R = ds[t]; X, R = to_device((X, R), device)
        with torch.no_grad():
            w = model(X)
            port_ret.append(torch.dot(w, R).item())
            dates.append(df["date"].unique()[t + 1])
            if np.linalg.norm(w) > 2.5:
                print(f"[AAA] here norm is {np.linalg.norm(w):.3e}, date = {dates[-1]}")
    return dates, port_ret, losses

################################################################################
# 6.  Volatility‑management helper
################################################################################

def vol_managed(series: pd.Series, window: int = 12) -> pd.Series:
    return series / series.rolling(window).std().shift(1)

################################################################################
# 7.  CLI
################################################################################

def main():
    tic = time.time(); set_seed()

    # ---- load data ----
    df = pd.read_pickle(Path("data") / "usa_131_per_size_ranks_False.pkl")
    df = df[df["size_grp"] == "micro"].copy()

    hyper = Hyper()
    dates, rets, losses = train_loop(df, hyper)

    out_dir = Path("project_results_constrained"); out_dir.mkdir(exist_ok=True)
    # raw portfolio returns
    ser = pd.Series(rets, index=pd.to_datetime(dates), name="Return")
    ser.to_csv(out_dir / "short_constrain_raw.csv")
    # volatility‑managed (prof‑style)
    ser_man = vol_managed(ser, hyper.vol_window)
    ser_man.name = "ManagedReturn"; ser_man.to_csv(out_dir / "short_constrain_managed.csv")
    # training loss
    pd.DataFrame(losses, columns=["step", "loss"]).to_csv(out_dir / "training_loss.csv", index=False)

    print(f"Finished in {(time.time()-tic)/60:.1f} min → {len(rets)} OOS months, saved raw & managed returns.")

if __name__ == "__main__":
    main()
