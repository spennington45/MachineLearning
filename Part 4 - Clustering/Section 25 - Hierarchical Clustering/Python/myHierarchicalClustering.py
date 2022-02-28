import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 2:].values

dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

hc = AgglomerativeClustering(n_clusters = 6, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

ax = plt.axes(projection='3d')
ax.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], X[y_hc == 0, 2], s = 100, c = 'red', label = 'Cluster 1')
ax.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], X[y_hc == 1, 2], s = 100, c = 'blue', label = 'Cluster 2')
ax.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], X[y_hc == 2, 2], s = 100, c = 'green', label = 'Cluster 3')
ax.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], X[y_hc == 3, 2], s = 100, c = 'cyan', label = 'Cluster 4')
ax.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], X[y_hc == 4, 2], s = 100, c = 'magenta', label = 'Cluster 5')
ax.scatter(X[y_hc == 5, 0], X[y_hc == 5, 1], X[y_hc == 5, 2], s = 100, c = 'yellow', label = 'Cluster 6')
plt.title('Clusters of customers')
plt.xlabel('Age (years)')
plt.ylabel('Annual Income (k$)')
# plt.zlabel("Spending Score (1-100)")
plt.legend()
plt.show()