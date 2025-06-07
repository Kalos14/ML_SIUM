# Optimized random-feature portfolio – **v2.0-cluster (20 epoche, ridge, no ReLU constraint, I/O optim.)**
# ---------------------------------------------------------------------
# • Rimuove qualunque ReLU sul vettore w (long/short unconstrained).
# • Ridge penalty impostato a 1.0 per penalizzare leva elevata.
# • Epoche portate a 20.
# • Pre-load mensile in month_tensors, usa DataLoader con num_workers.
# • Stampa warning se la leva gross (∑|w|) supera 1.5.
# ---------------------------------------------------------------------

import os, time, random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# --- Helper utils ---

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def num_workers_auto(max_w=8):
    cores = os.cpu_count() or 2
    return min(max_w, max(1, cores // 2))

# --- Random feature generation ---

def features_maker_prof(X: np.ndarray, G: int, P: int) -> pd.DataFrame:
    d = X.shape[1]
    S_hat_list = []
    for g in range(G):
        W_g = np.sqrt(2) * np.random.randn(d, P//(2*G)) / np.sqrt(d)
        XWg = X @ W_g
        cos_part = np.sqrt(2) * np.cos(XWg)
        sin_part = np.sqrt(2) * np.sin(XWg)
        S_hat_list.append(np.concatenate([cos_part, sin_part], axis=1))
    S_hat = np.concatenate(S_hat_list, axis=1)
    return pd.DataFrame(S_hat)

# --- Transformer model ---

class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__(); assert D % H == 0
        self.H, self.dk = H, D // H
        self.Wqkv = nn.Linear(D, 2*D, bias=False)
        self.scale = self.dk**0.5
    def forward(self, X):
        qk, v = self.Wqkv(X).chunk(2, -1)
        qk = qk.view(X.size(0), self.H, self.dk)
        v  = v.view(X.size(0), self.H, self.dk)
        att = torch.softmax(torch.einsum("nhd,mhd->hnm", qk, qk)/self.scale, -1)
        return torch.einsum("hnm,mhd->nhd", att, v).reshape(X.size(0), -1)

class FeedForward(nn.Module):
    def __init__(self, D, dF): super().__init__(); self.net = nn.Sequential(nn.Linear(D,dF), nn.ReLU(), nn.Linear(dF,D), nn.Dropout(0.1))
    def forward(self, X): return self.net(X)

class TransformerBlock(nn.Module):
    def __init__(self, D, H, dF):
        super().__init__(); self.attn=MultiHeadAttention(D,H); self.ffn=FeedForward(D,dF);
        self.l1=nn.LayerNorm(D); self.l2=nn.LayerNorm(D)
    def forward(self, X): return self.l2(X + self.ffn(self.l1(X + self.attn(X))))

class NonlinearPortfolioForward(nn.Module):
    def __init__(self, D, K, H=1, dF=256):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(D,H,dF) for _ in range(K)])
        self.lambda_out = nn.Parameter(torch.randn(D,1)/1000)
    def forward(self, X):
        for block in self.blocks: X = block(X)
        return (X @ self.lambda_out).squeeze()  # no ReLU

# --- Dataset for preloaded month tensors ---

class MonthTensorDataset(Dataset):
    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor]]): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# --- Training & OOS with DataLoader ---
lev_dates = 0
def train_loop(month_tensors, dates, window, epochs, lr, ridge):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = month_tensors[0][0].shape[1]
    model = NonlinearPortfolioForward(D=D, K=10, H=1, dF=256).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    dataset = MonthTensorDataset(month_tensors)
    nw = num_workers_auto()
    def loader(idxs): return DataLoader(Subset(dataset, idxs), batch_size=None, shuffle=False, pin_memory=True, num_workers=nw)

    rets, oos_dates = [], []
    for t in range(window, len(dataset)-1):
        # in-sample training
        for _ in range(epochs):
            for X_cpu, R_cpu in loader(range(t-window, t)):
                X, R = X_cpu.to(device), R_cpu.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    w = model(X)
                    loss = (1 - torch.dot(w,R))**2 + ridge * w.pow(2).sum()
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        # OOS prediction
        X_cpu, R_cpu = month_tensors[t]
        X, R = X_cpu.to(device), R_cpu.to(device)
        with torch.no_grad():
            w = model(X)
            lever = w.abs().sum().item()
            if lever > 1.5:
                print(f"[WARN] excessive leverage={lever:.2f} at date {dates[t]}")
                lev_dates= lev_dates+1
            rets.append(torch.dot(w,R).item()); oos_dates.append(dates[t+1])
    return oos_dates, rets

# --- Main ---

def main():
    set_seed()
    home = Path(f"/home/{os.getenv('USER')}")
    df = pd.read_pickle(home/"usa_131_per_size_ranks_False.pkl")
    df = df[df['size_grp']=='micro']

    # random features expansion
    P, G = 2000, 10
    records = []  # (date, X_tensor, R_tensor)
    for date, grp in df.groupby('date', sort=True):
        X_np = grp.drop(columns=['size_grp','date','r_1','id']).values
        S_t = features_maker_prof(X_np, G, P)
        X_t = torch.tensor(S_t.values, dtype=torch.float32)
        R_t = torch.tensor(grp['r_1'].values, dtype=torch.float32)
        records.append((date, X_t, R_t))
    dates = [r[0] for r in records]
    month_tensors = [(r[1], r[2]) for r in records]

    # run training & OOS
    window, epochs, lr, ridge = 60, 20, 1e-4, 1.0
    oos_dates, rets = train_loop(month_tensors, dates, window, epochs, lr, ridge)

    # save returns
    out = Path("project_results_ran_feat"); out.mkdir(exist_ok=True)
    pd.DataFrame({'Date':oos_dates,'Return':rets}).to_csv(out/"ranfeat_returns.csv", index=False)

    print("Done: epochs=20, ridge=1.0. Results in", out/"ranfeat_returns.csv")

if __name__=="__main__": main()
