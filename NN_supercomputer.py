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
output_dir = "NN_output"
os.makedirs(output_dir, exist_ok=True)


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

def train_loader(signals, returns):
  """
  This is a special DataLoader designed to work with portfolio optimization.
  It creates mini-batches using every month of data
  """
  dates = signals.index.get_level_values('date')
  unique_dates = dates.unique()
  for date in unique_dates:
    x = torch.tensor(signals.loc[dates == date].values, dtype=torch.float32).to(device)
    y = torch.tensor(returns.loc[dates == date].values, dtype=torch.float32).view(-1, 1).to(device)
    yield x, y

import numpy as np

class FlexibleMLP(nn.Module):
    def __init__(self, layers: list, scale: float=1.):
        """
        param: layers = list of integers
        """
        super(FlexibleMLP, self).__init__()

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i+1])

            # LeCun initialization
            nn.init.normal_(layer.weight, mean=0.0, std=scale * np.sqrt(1 / layers[i]))
            nn.init.normal_(layer.bias, mean=0.0, std=0 * np.sqrt(1 / layers[i]))

            self.layers.append(layer)
            # Add ReLU activation after each layer except the last
            if i < len(layers) - 2:
                self.activations.append(nn.ReLU())
            else:
                # Placeholder for the last layer's activation
                self.activations.append(nn.Identity())

    def forward(self, x, return_last_hidden=False):
        last_hidden = None

        for layer, activation in zip(self.layers[:-1], self.activations[:-1]):
            x = activation(layer(x))
            last_hidden = x  # Update last_hidden at each hidden layer

        # Apply the last layer without ReLU (or Identity for the placeholder)
        x = self.layers[-1](x)

        if return_last_hidden:
            return x, last_hidden
        return x

import random
def mssr_loss(output, target):
  """
  MSRR = Maximal Sharpe Ratio Regression
  This is our MSRR loss through which we evaluate the quality of predictions
  Every mini batch is a month. So,
  (output * target.view((output.shape[0], 1))).sum() is the return on the
  portfolio in that particular month (.sum() is over stocks)
  """
  dist = 1 - (output * target.view((output.shape[0], 1))).sum()
  msrr = torch.pow(dist, 2)
  return msrr

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)  # Set NumPy seed
    torch.manual_seed(seed_value)  # Set PyTorch seed
    random.seed(seed_value)  # Set Python random seed

    # If you are using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#OPTIONAL: Neural Netweork with rolling window (the only difference is that we evaluate the model at the end of each rolling window
# instead of at the end of the whole training)

ridge_penalty = 0.01
set_seed(42)



set_seed(0)  # Fixing the seed
num_epochs = 5
rolling_window = 120

stock_data = stock_data.loc[stock_data.index.get_level_values('date') >= '1990-01-01']

signals = stock_data.drop(columns=['r_1'])
labels = stock_data['r_1']

signals_rw = signals.copy()
labels_rw = labels.copy()

unique_dates = signals_rw.index.get_level_values(1).unique().sort_values()
results = []

device = "cuda" if torch.cuda.is_available() else "cpu"

for t in range(rolling_window, len(unique_dates)):

    model = FlexibleMLP([signals.shape[1],128,256,64, 1], scale=1.) # re-initializing weights !!!
    model.to(device)
    criterion = mssr_loss # this is our custom loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    train_dates = unique_dates[t - rolling_window:t]
    test_date = unique_dates[t] 

    train_signals = signals_rw.loc[pd.IndexSlice[:, train_dates], :]
    train_labels = labels_rw.loc[pd.IndexSlice[:, train_dates]]

    test_signals = signals_rw.loc[pd.IndexSlice[:, test_date], :]
    test_labels = labels_rw.loc[pd.IndexSlice[:, test_date]]

    for epoch in range(num_epochs):
        for inputs, targets in train_loader(train_signals, train_labels):
            # each mini batch is a month of data
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets) # this is (1- portfolio return)^2
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    model.eval()
    # test_data_predictions = model(torch.tensor(test_signals.values))
    # managed_returns = build_managed_returns(returns=test_labels, 
    #                                         signals=pd.DataFrame(test_data_predictions.detach().numpy(), index=test_signals.index)
    # )
    with torch.no_grad():
        test_tensor = torch.tensor(test_signals.values, dtype=torch.float32).to(device)
        test_data_predictions = model(test_tensor).cpu()  # Move to CPU if using GPU
    
    pred_df = pd.DataFrame(test_data_predictions.numpy(), index=test_signals.index)

    managed_returns = build_managed_returns(
        returns=test_labels,
        signals=pred_df
    )
    results.append(managed_returns)

#calculate sharpe ratio and save the results
results_df = pd.concat(results)
results_df.index = pd.to_datetime(results_df.index)
results_df.columns = ['returns']
sharpe_ratio_value = sharpe_ratio(results_df['returns'])
results_df.to_csv(f"{output_dir}/nn_rolling_window_results.csv")

# Plot cumulative returns and save the plot
(results_df / results_df.std()).cumsum().plot(title='Cumulative Returns of Neural Network OOS', figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Cumulative Returns (Standardized)')
plt.grid()
plt.savefig(f"{output_dir}/nn_rolling_window_cumulative_returns.png")
plt.close()

