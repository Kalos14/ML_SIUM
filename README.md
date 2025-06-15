# Machine Learning for Finance Project

In order to clarify as much as possible the structure of how all different
files are organized, the following README is provided.

Before we introduce the description of how the source code is provided,
it is important to say that for time and power constraint, the models were 
trained in the EPFL cluster, we did not train locally the full models but just a 
"test" version. However, the hyperparamters you will see in this source code, are 
meant to be the optimal ones, which indeed were only used when running 
on the cluster. For more details and specifications on how the code we run on the cluster 
looks like, please refer to our github https://github.com/Kalos14/ML_SIUM . 



The source code is splitted into two directories: "data" and "models".

## Data
This directory contains the .py from which we retrieved our dataset.
This is the only dataset we used to train our models, we downloaded it
via WRDS API, cleaned it and normalized all in the file "download_preprocesses_data.py".

For more insights on the motivations that brought to such cleaning and normalization, pleaser
refer to the report.

## Models

In our study we tried to apply two different models on the same 
dataset to develop trading strategies. For clarity, we thought it was better
to present the two different frameworks in two separate directories

### Ridge_and_RanFeat

In this directory you can find all the files regarding our implemenation of Ridge regression
and random feature to create optimal portfolio weights.

In each file, at the beginning, there is a more detailed desciption of the purposes
of the code. Here is a very brief summary of each file:

- main_ridge.py : implmentation of ridge regression on a rolling window to predict optimal 
portfolio. Test improvement of the strategy in function of increasing number of radom features
- benchmark_ridge.py : again applies the same model as before, but instead of increasing complexity
via random features, focus on few "known" significant factors
- functions_ridge.py : includes the functions that indeed define our model
- /pictures : stores our output

### Transformer

In this directory you can find all the files regarding our implemenation of the Transformer model to create optimal portfolio weights.

In each file, at the beginning, there is a more detailed desciption of the purposes
of the code. Here is a very brief summary of each file:

- main_file.py :  implementation of a Transformer Neural Network to predict optimal portfolio weights via a rolling-window training. The strategy is tested under three different constraint settings (vanilla, no short-selling, leverage-limited).
- functions_file.py: include all the functions that we used in our study
- volatility_manager.py: what we used to produce the volatility managed returns
- plotsaver.py: a usefull script that we used to plot our results 
