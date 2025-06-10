import pandas as pd
import datetime as dt
import wrds

#################################################################################################
# 1) Download raw dataset from WRDS - JKP characteristics
#################################################################################################

# Connect to WRDS
db = wrds.Connection()  

# Load the JKP characteristics list (abr_jkp column only)
chars = pd.read_excel('https://github.com/bkelly-lab/ReplicationCrisis/raw/master/GlobalFactors/Factor%20Details.xlsx')
chars_rel = chars[chars['abr_jkp'].notna()]['abr_jkp'].tolist()
chars_rel

# Define the SQL query to fetch data from the WRDS database
start_date = '19630131'
end_date = '20241231'

sql_query = f"""
    SELECT eom, permno, size_grp, me, ret, ret_exc_lead1m,{', '.join(chars_rel)},  sic, ff49
    FROM contrib.global_factor
    WHERE common = 1 
      AND exch_main = 1 
      AND primary_sec = 1 
      AND obs_main = 1 
      AND excntry = 'USA'
      AND eom >= '{start_date}'
      AND eom <= '{end_date}'
    ORDER BY permno, eom
"""

# Download the data
df = db.raw_sql(sql_query)

#################################################################################################
# 2) Preprocess the data
#################################################################################################

# Helper functions

def drop_col(df, threshold_max_na):
    """
    Drops inplace the columns of a dataframe who have more than "threshold" percent of Nan
    """
    return df.loc[:,df.isna().mean(axis = 0)<threshold_max_na]

def remove_rows_with_nan_threshold(df, threshold):
    """
    Remove rows from a DataFrame that have more than the specified threshold of NaN values.
    
    """
    # Calculate minimum number of non-NaN values required per row
    min_non_nan = int((1 - threshold) * len(df.columns))
    
    # Use dropna with thresh parameter
    return df.dropna(thresh=min_non_nan)

def rank_norm(group):
    return group.rank(axis=0, method='average', pct=True) - 0.5

# Drop stocks of size "nano"
df_clean = df[df["size_grp"] != "nano"] 

# Drop columns with more than 34% NaN values
df_clean = drop_col(df_clean, 0.34)

# Drop rows with more than 30% NaN values
df_clean = remove_rows_with_nan_threshold(df_clean, 0.3)

# Drop useless clumns for the analysis
df_clean = df_clean.drop(columns=["sic", "ff49", "me", "ret"])

# Rename columns for easier interpretation
df_clean.rename(columns = {"ret_exc_lead1m": "r_1", "eom": "date", "permno": "id"}, inplace = True)

# Drop rows with missing returns or identifiers
df_clean.dropna(subset = ["r_1"], inplace = True)
df_clean.dropna(subset = ["id"], inplace = True)

# Rank-normalize the characteristics month by month
grouped = df_clean.drop(columns = ["id", "r_1", "size_grp"]).groupby("date").transform(rank_norm)
df_final = pd.concat([df_clean[["date", "id", "size_grp", "r_1"]], grouped], axis=1)

# Fill NaN values with the median of each column (i.e., 0 after rank-normalization)
df_final.fillna(0, inplace = True)

# Sort the final DataFrame first by date then id
df_final = df_final.sort_values(by=["date", "id"]).reset_index(drop=True)

# Save the final DataFrame to a pickle file
df_final.to_pickle("our_version_norm.pkl")