import pandas as pd
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

N = 10000 # number of users
d = 10 # number of ads
adsSelected = []
numberOfSelections = [0] * d # Ni(n)
sumOfRewards = [0] * d # Ri(n)
totalRewards = 0

for n in range(0, N):
    selected = 0
    maxUpperBount = 0
    for ad in range(0, d):
        if (numberOfSelections[ad] > 0):
            averageReward = sumOfRewards[ad] / numberOfSelections[ad]
            deltaI = math.sqrt(3/2 * math.log(n + 1) / numberOfSelections[ad])
            upperBount = averageReward + deltaI
        else:
            upperBount = 1e400
        if (upperBount > maxUpperBount):
            maxUpperBount = upperBount
            selected = ad
    adsSelected.append(selected)
    numberOfSelections[selected] += 1
    reward = dataset.values[n, selected]
    sumOfRewards[selected] += reward
    totalRewards += reward

plt.hist(adsSelected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
    
