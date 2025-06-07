# Optimized portfolio transformer – **v1.8‑cluster (20 epoche, path assoluti)**
# ---------------------------------------------------------------------
#   • Identico a v1.8 ma con path hard‑coded in $HOME per i CSV su cluster.
#   • Legge:
#       /home/$USER/usa_131_per_size_ranks_False.pkl
#       /home/$USER/SandP benchmark.csv
#   • Se i file non esistono, fallback a $DATA_DIR o ./data.
# ---------------------------------------------------------------------

import os, time, sys
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

################################################################################
# Helper
################################################################################

def set_seed(seed: int = 42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

def to_device(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev, non_blocking=True)
    return tuple(o.to(dev, non_blocking=True) for o in obj)

def num_workers_auto(max_w=8):
    cores = os.cpu_count() or 2
    return min(max_w, max(1, cores // 2))

################################################################################
# Hyper
################################################################################
@dataclass
class Hyper:
    window: int = 60
    epochs: int = 20
    K: int = 10; H: int = 1; dF: int = 256
    lr: float = 1e-5
    ridge: float = 1.0
    vol_window: int = 12
    use_compile: bool = False

################################################################################
# Dataset
################################################################################
class MonthlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feat: List[str]):
        self.months = [m for _, m in df.groupby("date", sort=True)]
        self.feat = feat
    def __len__(self): return len(self.months)
    def __getitem__(self, idx):
        g = self.months[idx]
        X = torch.tensor(g[self.feat].values, dtype=torch.float32)
        R = torch.tensor(g["r_1"].values, dtype=torch.float32)
        return X, R

################################################################################
# Model (no L1 budget, leva variabile)
################################################################################
class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__(); assert D % H == 0
        self.H, self.dk = H, D // H
        self.Wqkv = nn.Linear(D, 2*D, bias=False)
        self.scale = self.dk ** 0.5
    def forward(self, X):
        qk, v = self.Wqkv(X).chunk(2, -1)
        qk = qk.view(X.size(0), self.H, self.dk); v = v.view(X.size(0), self.H, self.dk)
        a = torch.softmax(torch.einsum("nhd,mhd->hnm", qk, qk)/self.scale, -1)
        return torch.einsum("hnm,mhd->nhd", a, v).reshape(X.size(0), -1)

class FeedForward(nn.Module):
    def __init__(self, D, dF):
        super().__init__(); self.net = nn.Sequential(nn.Linear(D,dF), nn.ReLU(), nn.Linear(dF,D), nn.Dropout(0.1))
    def forward(self, X): return self.net(X)

class TransformerBlock(nn.Module):
    def __init__(self, D, H, dF):
        super().__init__(); self.attn = MultiHeadAttention(D,H); self.ffn = FeedForward(D,dF); self.l1=nn.LayerNorm(D); self.l2=nn.LayerNorm(D)
    def forward(self, X):
        X = self.l1(X + self.attn(X)); return self.l2(X + self.ffn(X))

class PortfolioTransformer(nn.Module):
    def __init__(self, D, hyp: Hyper):
        super().__init__(); self.blocks = nn.Sequential(*[TransformerBlock(D,hyp.H,hyp.dF) for _ in range(hyp.K)]); self.out = nn.Linear(D,1,bias=False)
    def forward(self, X):
        return self.out(self.blocks(X)).flatten()

################################################################################
# Training / OOS
################################################################################

def train_loop(df: pd.DataFrame, hyp: Hyper):
    feat = [c for c in df.columns if c not in {"size_grp","date","r_1","id"}]
    ds = MonthlyDataset(df, feat)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PortfolioTransformer(len(feat), hyp).to(dev)
    opt = optim.Adam(model.parameters(), lr=hyp.lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available(), device_type="cuda")

    nw = num_workers_auto()
    make_loader = lambda idxs: DataLoader(Subset(ds, idxs), batch_size=None, shuffle=False, pin_memory=True, num_workers=nw)

    rets, dates, losses, step = [], [], [], 0
    uq_dates = df["date"].unique()

    for t in range(hyp.window, len(ds)-1):
        loader = make_loader(range(t-hyp.window, t))
        for _ in range(hyp.epochs):
            for X,R in loader:
                X,R = to_device((X,R), dev)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    w = model(X); loss = (1 - torch.dot(w,R))**2 + hyp.ridge * w.pow(2).sum()
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                losses.append((step, loss.item())); step += 1
        # OOS
        X,R = ds[t]; X,R = to_device((X,R), dev)
        with torch.no_grad():
            w = model(X); rets.append(torch.dot(w,R).item()); dates.append(uq_dates[t+1])
    return dates, rets, losses

################################################################################
# Vol‑managed
################################################################################

def vol_managed(series: pd.Series, window:int):
    return series / series.rolling(window).std().shift(1)

################################################################################
# Main
################################################################################

def main():
    set_seed(); start = time.time()

    # ----- absolute paths on cluster -----
    home = Path(f"/home/{os.getenv('USER')}")
    dataset_path  = home/"usa_131_per_size_ranks_False.pkl"
    bench_path    = home/"SandP benchmark.csv"

    if dataset_path.exists():
        df = pd.read_pickle(dataset_path)
    else:
        # fallback
        df = pd.read_pickle(Path(os.getenv("DATA_DIR","data"))/"usa_131_per_size_ranks_False.pkl")
    df = df.query("size_grp=='micro'")

    hyp = Hyper()
    dates, rets, losses = train_loop(df, hyp)

    out = Path("project_results_constrained"); out.mkdir(exist_ok=True)

    ser = pd.Series(rets, index=pd.to_datetime(dates), name="Return")
    ser.to_csv(out/"short_constrain_raw.csv")
    ser_man = vol_managed(ser, hyp.vol_window).rename("ManagedReturn")
    ser_man.to_csv(out/"short_constrain_managed.csv")
    pd.DataFrame(losses, columns=["step","loss"]).to_csv(out/"training_loss.csv", index=False)

    # Sharpe
    sharpe = lambda s: (s.mean()/s.std(ddof=0))*np.sqrt(12)
    sr_raw, sr_man = sharpe(ser), sharpe(ser_man)

    # Benchmark plot
    if bench_path.exists():
        bench = pd.read_csv(bench_path, parse_dates=[0])
        bench.set_index(bench.columns[0], inplace=True)
        bench_series = bench[bench.columns[-1]].rename("Benchmark") / 100
        bench_series = bench_series.reindex(ser.index).dropna()
        s_al, b_al = ser.align(bench_series, join="inner")
        plt.figure(figsize=(9,5))
        plt.plot(s_al.cumsum(), label="Raw cumsum")
        plt.plot(b_al.cumsum(), label="S&P cumsum", linestyle="--")
        plt.title(f"CumSum Raw vs S&P | SR_raw {sr_raw:.2f}")
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=10))
        plt.gca().x
