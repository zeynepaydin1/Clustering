import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import preprocessing

time_matrix = pd.DataFrame(preprocessing.preprocess("djlorj.txt"))
pca = PCA(n_components=2)
points = pca.fit_transform(time_matrix)

dbscan = DBSCAN(eps=0.5, min_samples=2)
labels = dbscan.fit_predict(points)

plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.show()
