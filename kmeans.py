import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import preprocessing

time_matrix = pd.DataFrame(preprocessing.preprocess("djlorj.txt"))

#reduce the dimensionality of the matrix
pca = PCA(n_components=2)  #for visualization
points = pca.fit_transform(time_matrix)

k_values = range(1, 10)
inertia = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(points)
    inertia.append(kmeans.inertia_)  #within-cluster sum of squares

plt.plot(k_values, inertia, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

optimal_k = 4  #based on the elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(points)

plt.scatter(points[:, 0], points[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.title(f'K-means Clustering with k={optimal_k}')
plt.legend()
plt.show()
