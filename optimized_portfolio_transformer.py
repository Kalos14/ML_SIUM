# Optimized random-feature portfolio – **v2.1-cluster (plot + Sharpe)**
# ---------------------------------------------------------------------
# • V2.1: aggiunto calcolo e plot Sharpe ratio sui returns OOS.
# • Salva anche il grafico cumulative-return con Sharpe in title.
# • Mantiene tutte le feature di v2.0 (20 epoche, ridge=1.0, no ReLU,
#   I/O optim., warning leva).
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def features_maker_prof(X: np.ndarray, G: int, P: int) -> np.ndarray:
    d = X.shape[1]
    S_parts = []
    for g in range(G):
        Wg = np.sqrt(2) * np.random.randn(d, P//(2*G)) / np.sqrt(d)
        XW = X @ Wg
        S_parts.append(np.sqrt(2)*np.cos(XW))
        S_parts.append(np.sqrt(2)*np.sin(XW))
    return np.concatenate(S_parts, axis=1)

# --- Model ---

class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__(); assert D % H == 0
        self.H, self.dk = H, D//H
        self.Wqkv = nn.Linear(D,2*D,bias=False)
        self.scale = self.dk**0.5
    def forward(self,X):
        qk,v = self.Wqkv(X).chunk(2,-1)
        qk = qk.view(X.size(0),self.H,self.dk)
        v  = v.view(X.size(0),self.H,self.dk)
        att = torch.softmax(torch.einsum("nhd,mhd->hnm",qk,qk)/self.scale,-1)
        return torch.einsum("hnm,mhd->nhd",att,v).reshape(X.size(0),-1)

class FeedForward(nn.Module):
    def __init__(self,D,dF): super().__init__(); self.net=nn.Sequential(nn.Linear(D,dF),nn.ReLU(),nn.Linear(dF,D),nn.Dropout(0.1))
    def forward(self,X): return self.net(X)

class TransformerBlock(nn.Module):
    def __init__(self,D,H,dF):
        super().__init__(); self.att=MultiHeadAttention(D,H); self.ff=FeedForward(D,dF)
        self.ln1=nn.LayerNorm(D); self.ln2=nn.LayerNorm(D)
    def forward(self,X): return self.ln2(self.ln1(X+self.att(X))+self.ff(self.ln1(X+self.att(X))))

class NonlinearPortfolioForward(nn.Module):
    def __init__(self,D,K,H=1,dF=256):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(D,H,dF) for _ in range(K)])
        self.lambda_out = nn.Parameter(torch.randn(D,1)/1000)
    def forward(self,X):
        for b in self.blocks: X = b(X)
        return (X @ self.lambda_out).squeeze()

# --- Dataset ---

class MonthTensorDataset(Dataset):
    def __init__(self,data:List[Tuple[torch.Tensor,torch.Tensor]]): self.data=data
    def __len__(self): return len(self.data)
    def __getitem__(self,idx): return self.data[idx]

# --- Training loop ---

def train_loop(month_tensors, dates, window, epochs, lr, ridge):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = month_tensors[0][0].shape[1]
    model = NonlinearPortfolioForward(D, K=10, H=1, dF=256).to(device)
    opt = optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    dataset = MonthTensorDataset(month_tensors)
    nw = num_workers_auto()
    def loader(idxs):
        return DataLoader(Subset(dataset,idxs),batch_size=None,shuffle=False,pin_memory=True,num_workers=nw)

    rets, oos_dates = [],[]
    for t in range(window, len(dataset)-1):
        # in-sample
        for _ in range(epochs):
            for Xc,Rc in loader(range(t-window,t)):
                X,R = Xc.to(device), Rc.to(device)
                with torch.amp.autocast(enabled=torch.cuda.is_available()):
                    w = model(X)
                    loss = (1-torch.dot(w,R))**2 + ridge * w.pow(2).sum()
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        # OOS
        Xc,Rc = month_tensors[t]
        X,R = Xc.to(device), Rc.to(device)
        with torch.no_grad():
            w = model(X)
            lever = w.abs().sum().item()
            if lever>1.5:
                print(f"[WARN] excessive leverage={lever:.2f} at {dates[t]}")
            rets.append(torch.dot(w,R).item()); oos_dates.append(dates[t+1])
    return oos_dates,rets

# --- Main ---

def main():
    set_seed()
    home = Path(f"/home/{os.getenv('USER')}")
    df = pd.read_pickle(home/"usa_131_per_size_ranks_False.pkl")
    df = df[df['size_grp']=='micro']

    # random features
    P,G = 2000,10
    records=[]
    for date,grp in df.groupby('date',sort=True):
        Xn=grp.drop(columns=['size_grp','date','r_1','id']).values
        S = features_maker_prof(Xn,G,P)
        records.append((date,torch.tensor(S,dtype=torch.float32),torch.tensor(grp['r_1'].values,dtype=torch.float32)))
    dates = [r[0] for r in records]
    month_tensors = [(r[1],r[2]) for r in records]

    # train & OOS
    window,epochs,lr,ridge = 60,20,1e-4,1.0
    oos_dates,rets = train_loop(month_tensors,dates,window,epochs,lr,ridge)

    # save returns
    out=Path("project_results_ran_feat"); out.mkdir(exist_ok=True)
    df_ret=pd.DataFrame({'Date':oos_dates,'Return':rets}); df_ret.to_csv(out/"ranfeat_returns.csv",index=False)

    # compute Sharpe
    arr = np.array(rets)
    sr = (arr.mean()/arr.std(ddof=0))*np.sqrt(12)

    # plot cumulative returns
    dates_pd = pd.to_datetime(oos_dates)
    cum = np.cumsum(arr)
    plt.figure(figsize=(10,5))
    plt.plot(dates_pd,cum, label="CumSum Return")
    plt.title(f"Random-Feature Portfolio | SR={sr:.2f}")
    plt.xlabel("Date"); plt.ylabel("CumSum Return")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out/"ranfeat_plot.png",dpi=150)
    plt.close()

    print(f"Done: epochs={epochs}, ridge={ridge}, Sharpe={sr:.2f}")
    print("Results in", out/"ranfeat_returns.csv", "and ranfeat_plot.png")

if __name__=="__main__": main()
