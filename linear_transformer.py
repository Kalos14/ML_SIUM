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
import matplotlib.pyplot as plt
import numpy as np
import os
### SUPERCOMPUTER ###
output_dir = "Ridge_results"
os.makedirs(output_dir, exist_ok=True)

print("you are running the code about Ridge regression")

dataset_path = f"/home/{os.environ['USER']}/usa_131_per_size_ranks_False.pkl"
stock_data = pd.read_pickle(dataset_path)

benchmark_path = f"/home/{os.environ['USER']}/SandP benchmark.csv"

SP_benchmark = pd.read_csv(benchmark_path)
SP_benchmark["caldt"] = pd.to_datetime(SP_benchmark["caldt"])

size_group = 'mega'
if size_group is not None:
  stock_data = stock_data.loc[stock_data.size_grp==size_group]

stock_data.set_index(["id", "date"], inplace=True)
size_groups = stock_data.pop('size_grp')
signals = stock_data.drop(columns=['r_1'])

def build_managed_returns(returns, signals):
  # I am using numpy broadcasting here
  managed_returns = (signals * returns.values.reshape(-1, 1)).groupby(signals.index.get_level_values('date')).sum()
  return managed_returns

managed_returns = build_managed_returns(returns = stock_data['r_1'], signals = signals)

def sharpe_ratio(returns):
  """
  The data is at monthly frequency, hence we multiply by sqrt(12)
  """
  return np.round(np.sqrt(12) * returns.mean() / returns.std(), 2)

def volatility_managed_returns(rets, window):
  return rets / rets.rolling(window).std().shift(1)

def ridge_regr(signals: np.ndarray,
                  labels: np.ndarray,
                  future_signals: np.ndarray,
                  shrinkage_list: np.ndarray):
    """
    Regression is
    beta = (zI + S'S/t)^{-1}S'y/t = S' (zI+SS'/t)^{-1}y/t
    Inverting matrices is costly, so we use eigenvalue decomposition:
    (zI+A)^{-1} = U (zI+D)^{-1} U' where UDU' = A is eigenvalue decomposition,
    and we use the fact that D @ B = (diag(D) * B) for diagonal D, which saves a lot of compute cost
    :param signals: S
    :param labels: y
    :param future_signals: out of sample y
    :param shrinkage_list: list of ridge parameters
    :return:
    """
    t_ = signals.shape[0]
    p_ = signals.shape[1]
    if p_ < t_:
        # this is standard regression
        eigenvalues, eigenvectors = np.linalg.eigh(signals.T @ signals / t_)
        means = signals.T @ labels.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means
        intermed = np.concatenate([(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in shrinkage_list],
                                  axis=1)
        betas = eigenvectors @ intermed
    else:
        # this is the weird over-parametrized regime
        eigenvalues, eigenvectors = np.linalg.eigh(signals @ signals.T / t_)
        means = labels.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means # this is \mu

        # now we build [(z_1+\delta)^{-1}, \cdots, (z_K+\delta)^{-1}] * \mu
        intermed = np.concatenate([(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in shrinkage_list],
                                  axis=1)

        tmp = eigenvectors.T @ signals # U.T @ S
        betas = tmp.T @ intermed # (S.T @ U) @ [(z_1+\delta)^{-1}, \cdots, (z_K+\delta)^{-1}] * \mu
    predictions = future_signals @ betas
    return betas, predictions

def hw_efficient_portfolio_oos(raw_factor_returns: pd.DataFrame, window: int = 120):
    """
    Computes out-of-sample (OOS) returns of a ridge-regularized portfolio.

    For each time point `t`, the function:
    - Trains a ridge regression model over a rolling window of the past 120 months.
    - Uses the model to predict weights that replicate an equally weighted target.
    - Applies the estimated weights to compute a one-period out-of-sample return.

    The function evaluates multiple shrinkage levels and reports their cumulative returns and Sharpe ratios.

    Parameters:
    ----------
    raw_factor_returns : pd.DataFrame
        A DataFrame of monthly factor returns with datetime indices.

    Returns:
    -------
    oos_df : pd.DataFrame
        A DataFrame where each column corresponds to a shrinkage value and contains the OOS returns over time.
    """
    oos_returns = []
    dates = []
    shrinkage_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]#[0.01, 0.1, 1, 10, 100]  

    for t in range(window,len(raw_factor_returns)):
        X_train = raw_factor_returns.iloc[t-120:t,:].values
        y_train = np.ones((X_train.shape[0], 1)) 
        X_test = raw_factor_returns.iloc[t,:].values
        beta, optimal = ridge_regr(signals=X_train,
                       labels=y_train,
                       future_signals=X_test,
                       shrinkage_list=shrinkage_list)
        oos_returns.append(optimal)
        dates.append(raw_factor_returns.index[t])
        #print(f'dimensione di y: {optimal.shape}, dimensione di beta: {beta.shape}')
    
    oos_df = pd.DataFrame(oos_returns, index=dates, columns=shrinkage_list)

    #save the results
    oos_df.to_csv(f"{output_dir}/ridge_oos_results.csv")
    # Plot cumulative returns for all shrinkage values and save the plot
    (oos_df / oos_df.std()).cumsum().plot(title='Cumulative Returns of Ridge Regression OOS', figsize=(12, 6))
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (Standardized)')
    plt.legend(title='Shrinkage Values')
    plt.grid()
    plt.savefig(f"{output_dir}/ridge_cumulative_returns.png")
    plt.close()
    #(oos_df / oos_df.std()).cumsum().plot()
    # Optionally, print Sharpe ratios
    # for s in shrinkage_list:
    #     print(f"Shrinkage {s:<10}: Sharpe {sharpe_ratio(oos_df[s]):.2f}")
    #calculate and save the Sharpe ratios
    sharpe_ratios = {s: sharpe_ratio(oos_df[s]) for s in shrinkage_list}
    sharpe_ratios_df = pd.DataFrame.from_dict(sharpe_ratios, orient='index', columns=['Sharpe Ratio'])
    sharpe_ratios_df.to_csv(f"{output_dir}/ridge_sharpe_ratios.csv")

    return oos_df
    
def produce_random_feature_managed_returns(P, stock_data, signals, num_seeds=2, scale=1.0):
  """
  Suppose I wanted to build P random features. Using the weights \Theta \in \R^{d\times P},
  I could just do signals @ \Theta. If signals are (NT)\times d dimensional, then
  signals @ \Theta are (NT) \times P dimensional.
  Inststead, we can generate random features in small chunks, compute factors and proceed further.
  """
  all_random_feature_managed_returns = pd.DataFrame()
  d = signals.shape[1]
  #print(d)
  for seed in range(num_seeds):
    # every seed gives me a new chunk of factors
    np.random.seed(seed)
    omega = scale * np.sqrt(2) * np.random.randn(P, d) / np.sqrt(d)
    ins_sin = np.sqrt(2) * np.sin(signals @ omega.T) # signals @ \Theta are (NT) \times P dimensional.
    ins_cos = np.sqrt(2) * np.cos(signals @ omega.T) # signals @ \Theta are (NT) \times P dimensional.
    random_features = pd.concat([ins_sin, ins_cos], axis=1)

    # Now, I collapse the N dimension.
    random_feature_managed_returns = build_managed_returns(returns=stock_data['r_1'], signals=random_features)
    # random_feature_managed_returns are now T \times P
    all_random_feature_managed_returns = pd.concat([all_random_feature_managed_returns, random_feature_managed_returns], axis=1)
  return all_random_feature_managed_returns

P =  18000#1000
#signals is stock data without the returns

d = signals.shape[1] # d=6 momentum signals
scale = 1.

hw_random_feature_managed_returns = produce_random_feature_managed_returns(P, stock_data, signals, num_seeds=10)
hw_optimal_random_features = hw_efficient_portfolio_oos(raw_factor_returns=hw_random_feature_managed_returns, window=360)


