import pandas as pd
from apyori import apriori

dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
# dataset.info()

transactions = []
# for i in range(0, len(dataset.index)):
#     transactions.append([str(dataset.values[i, j]for j in range(0, len(dataset.columns)))])
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.5, min_lift = 3)

results = list(rules)
# print(results)

# def inspect(results):
#     lhs         = [tuple(result[2][0][0])[0] for result in results]
#     rhs         = [tuple(result[2][0][1])[0] for result in results]
#     supports    = [result[1] for result in results]
#     confidences = [result[2][0][2] for result in results]
#     lifts       = [result[2][0][3] for result in results]
#     return list(zip(lhs, rhs, supports, confidences, lifts))
# resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

def inspect(results):
    lhs         = [result[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['items', 'Support', 'Confidence', 'Lift'])

print(resultsinDataFrame.nlargest(n = 100, columns = 'Lift'))
