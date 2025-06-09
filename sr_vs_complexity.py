import numpy as np
from scipy.stats import rankdata
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os



#################################SUPERCOMPUTER

dataset_path = f"/home/{os.environ['USER']}/our_version_norm.pkl"
stock_data = pd.read_pickle(dataset_path)

size_group = 'large'
if size_group is not None:
  stock_data = stock_data.loc[stock_data.size_grp==size_group]

stock_data.set_index(["id", "date"], inplace=True)
size_groups = stock_data.pop('size_grp')

def build_managed_returns(returns, signals):
  # I am using numpy broadcasting here
  managed_returns = (signals * returns.values.reshape(-1, 1)).groupby(signals.index.get_level_values('date')).sum()
  return managed_returns

signals = stock_data.drop(columns=['r_1'])
hw_managed_returns = build_managed_returns(returns = stock_data['r_1'], signals = signals)

def sharpe_ratio(returns):
  """
  The data is at monthly frequency, hence we multiply by sqrt(12)
  """
  return np.round(np.sqrt(12) * returns.mean() / returns.std(), 2)

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

def hw_efficient_portfolio_oos(raw_factor_returns: pd.DataFrame, P: int, shrinkage_list=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]):
    oos_returns = []
    dates = []

    for t in range(360, len(raw_factor_returns)):
        X_train = raw_factor_returns.iloc[t-360:t, :].values
        y_train = np.ones((X_train.shape[0], 1)) 
        X_test = raw_factor_returns.iloc[t, :].values

        beta, optimal = ridge_regr(signals=X_train,
                                   labels=y_train,
                                   future_signals=X_test,
                                   shrinkage_list=shrinkage_list)
        oos_returns.append(optimal)
        dates.append(raw_factor_returns.index[t])
    
    oos_df = pd.DataFrame(oos_returns, index=dates, columns=shrinkage_list)

    sharpe_dict = {s: sharpe_ratio(oos_df[s]) for s in shrinkage_list}
    for s, sr in sharpe_dict.items():
        print(f"P = {P} -> Sharpe = {sr:.2f}")

    return oos_df, sharpe_dict

def produce_random_feature_managed_returns(P, stock_data, signals, num_seeds=10, scale = 1.0):
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



# Local output directory
output_dir = "plot_ridge_results_low_penalties"
os.makedirs(output_dir, exist_ok=True)

# Define complexity levels
complexities = [1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
sharpe_by_P = {}     # {P: {λ: Sharpe}}
returns_by_P = {}    # {(P, λ): return_series}

# Loop over complexity levels
for P in complexities:
    print(f"Processing P = {P}...")
    hw_random_feature_managed_returns = produce_random_feature_managed_returns(P, stock_data, signals)
    oos_df, sharpe_dict = hw_efficient_portfolio_oos(hw_random_feature_managed_returns,P)

    sharpe_by_P[P] = sharpe_dict

    for shrink, series in oos_df.items():
        standardized = series / series.std()
        returns_by_P[(P, shrink)] = standardized

# Save Sharpe ratios to CSV
sharpe_df = pd.DataFrame.from_dict(sharpe_by_P, orient='index')  # rows = P, cols = λ
sharpe_df.index.name = "P"
sharpe_df.to_csv(os.path.join(output_dir, "sharpe_ratios_by_P_large_low_penalties.csv"))
print("Saved Sharpe ratios to CSV.")

# Save cumulative returns to CSV
returns_df = pd.DataFrame({
    f"P={P}_λ={shrink}": ret.cumsum()
    for (P, shrink), ret in returns_by_P.items()
})
returns_df.index.name = "Date"
returns_df.to_csv(os.path.join(output_dir, "cumulative_returns_by_P_large_low_penalties.csv"))
print("Saved cumulative returns to CSV.")

# Plot cumulative returns
plt.figure(figsize=(12, 6))
for label, cumret in returns_df.items():
    cumret.plot(label=label)

plt.title("Cumulative Returns by Complexity P and Shrinkage λ")
plt.xlabel("Date")
plt.ylabel("Cumulative Standardized Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "all_cumrets.png"))
plt.close()
print("Saved cumulative returns plot.")

# Plot Sharpe ratios vs P for each shrinkage level
plt.figure(figsize=(8, 5))
for shrink in sharpe_df.columns:
    plt.plot(sharpe_df.index, sharpe_df[shrink], marker='o', label=f"λ={shrink}")

plt.title("Sharpe Ratio vs Complexity P")
plt.xlabel("Number of Random Features (P)")
plt.ylabel("Sharpe Ratio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sharpe_vs_P_mega.png"))
plt.close()
print("Saved Sharpe ratio plot.")





















#CLUSTER
# complexities = [1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# sharpe_by_P = {}     # {P: {λ: Sharpe}}
# returns_by_P = {}    # {(P, λ): return_series}

# # Loop over complexity levels
# for P in complexities:
#     hw_random_feature_managed_returns = produce_random_feature_managed_returns(P, stock_data, signals)
#     oos_df, sharpe_dict = hw_efficient_portfolio_oos(hw_random_feature_managed_returns)

#     sharpe_by_P[P] = sharpe_dict

#     for shrink, series in oos_df.items():
#         standardized = series / series.std()
#         returns_by_P[(P, shrink)] = standardized

# # Save Sharpe ratios to CSV
# sharpe_df = pd.DataFrame.from_dict(sharpe_by_P, orient='index')  # rows = P, cols = λ
# sharpe_df.index.name = "P"
# sharpe_df.to_csv(os.path.join(output_dir, "sharpe_ratios_by_P.csv"))

# # Save cumulative returns to CSV
# returns_df = pd.DataFrame({
#     f"P={P}_λ={shrink}": ret.cumsum()
#     for (P, shrink), ret in returns_by_P.items()
# })
# returns_df.index.name = "Date"
# returns_df.to_csv(os.path.join(output_dir, "cumulative_returns_by_P.csv"))

# # Plot cumulative returns
# plt.figure(figsize=(12, 6))
# for label, cumret in returns_df.items():
#     cumret.plot(label=label)

# plt.title("Cumulative Returns by Complexity P and Shrinkage λ")
# plt.xlabel("Date")
# plt.ylabel("Cumulative Standardized Return")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "all_cumrets.png"))
# plt.close()

# # Plot Sharpe ratios vs P for each shrinkage level
# plt.figure(figsize=(8, 5))
# for shrink in sharpe_df.columns:
#     plt.plot(sharpe_df.index, sharpe_df[shrink], marker='o', label=f"λ={shrink}")

# plt.title("Sharpe Ratio vs Complexity P")
# plt.xlabel("Number of Random Features (P)")
# plt.ylabel("Sharpe Ratio")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "sharpe_vs_P.png"))
# plt.close()






























################single ridge penalty###########################
# complexities = [1, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# sharpe_by_P = {}
# returns_by_P = {}

# for P in complexities:
#     hw_random_feature_managed_returns = produce_random_feature_managed_returns(P, stock_data, signals)
#     oos_df, sharpe_dict = hw_efficient_portfolio_oos(hw_random_feature_managed_returns,P)
    
#     # Assuming only one shrinkage value
#     sharpe_by_P[P] = list(sharpe_dict.values())[0]
#     returns_by_P[P] = (oos_df / oos_df.std()).squeeze()  # standardized cumulative return

# # Save Sharpe ratios to CSV
# sharpe_df = pd.DataFrame.from_dict(sharpe_by_P, orient='index', columns=['Sharpe'])
# sharpe_df.index.name = "P"
# sharpe_csv_path = os.path.join(output_dir, "sharpe_ratios_by_P.csv")
# sharpe_df.to_csv(sharpe_csv_path)

# # Save cumulative returns to CSV
# returns_df = pd.DataFrame({f"P={P}": ret.cumsum() for P, ret in returns_by_P.items()})
# returns_df.index.name = "Date"
# returns_csv_path = os.path.join(output_dir, "cumulative_returns_by_P.csv")
# returns_df.to_csv(returns_csv_path)


# plt.figure(figsize=(12, 6))
# for P, cumret in returns_by_P.items():
#     cumret.cumsum().plot(label=f"P={P}")

# plt.title("Cumulative Returns by Complexity P")
# plt.xlabel("Date")
# plt.ylabel("Cumulative Standardized Return")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "all_cumrets.png"))
# plt.close()

# plt.figure(figsize=(8, 5))
# plt.plot(list(sharpe_by_P.keys()), list(sharpe_by_P.values()), marker='o')
# plt.title("Sharpe Ratio vs Complexity P")
# plt.xlabel("Number of Random Features (P)")
# plt.ylabel("Sharpe Ratio")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "sharpe_vs_P.png"))
# plt.close()
