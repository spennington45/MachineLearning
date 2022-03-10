import pandas as pd

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
# dataset.info()

transaction = []
for i in range(0, len(dataset.index)):
    transaction.append([str(dataset.values[i, j]for j in range(0, len(dataset.columns)))])

