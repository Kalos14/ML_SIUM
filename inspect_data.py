
import numpy as np
from scipy.stats import rankdata  # Make sure this is included!
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime as dt
import os
import matplotlib.pyplot as plt

# In[2]:


setseed = 42
np.random.seed(setseed)
torch.manual_seed(setseed)




# Percorso al file (modifica se il file è altrove)
dataset_path = f"/home/{os.environ['USER']}/usa_131_per_size_ranks_False.pkl"

# Carica il dataset
try:
    df = pd.read_pickle(dataset_path)
except Exception as e:
    print(f"Errore nel caricamento del dataset: {e}")
    exit(1)

# Crea directory di output
output_dir = "results/inspect"
os.makedirs(output_dir, exist_ok=True)

# Percorso file di output
output_file = os.path.join(output_dir, "dataset_summary.txt")

# Scrive un riassunto del dataset su file
with open(output_file, "w") as f:
    f.write("==== HEAD ====\n")
    f.write(df.head().to_string())
    f.write("\n\n==== INFO ====\n")
    df.info(buf=f)
    f.write("\n\n==== DESCRIBE ====\n")
    f.write(df.describe().to_string())
    f.write("\n\n==== VALORI NULLI ====\n")
    f.write(df.isnull().sum().to_string())
    f.write("\n\n==== TIPI DI DATO ====\n")
    f.write(df.dtypes.to_string())

print(f"Analisi salvata in {output_file}")


# Esempio di grafico
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Esempio di grafico")

# Salva il file
plt.savefig(os.path.join(output_dir, "grafico1.png"))
plt.close()  # Libera memoria, importante nei cluster