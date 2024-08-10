from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

import preprocessing

distance_matrix_cleaned = preprocessing.preprocess("djlorj.txt")
distance_matrix = np.array(distance_matrix_cleaned)

sym_distance_matrix = (distance_matrix + distance_matrix.T) / 2

condensed_distance_matrix = pdist(sym_distance_matrix)

Z = linkage(condensed_distance_matrix, method='ward')

plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dendrogram(Z)
plt.show()

def plot_elbow_method(Z):
    last = Z[-10:, 2]
    last_rev = last[::-1]
    indexes = np.arange(1, len(last) + 1)
    plt.figure(figsize=(10, 7))
    plt.plot(indexes, last_rev)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

plot_elbow_method(Z)
num_clusters = 4

clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

print("Cluster assignments for each point:")
print(clusters)

clustered_points = {}
for idx, cluster_id in enumerate(clusters):
    if cluster_id not in clustered_points:
        clustered_points[cluster_id] = []
    clustered_points[cluster_id].append(idx)

for cluster_id, points in clustered_points.items():
    print(f"Cluster {cluster_id}: Points {points}")

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
points_2d = mds.fit_transform(sym_distance_matrix)

plt.figure(figsize=(10, 7))
for cluster_id in np.unique(clusters):
    cluster_points = points_2d[clusters == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}')
plt.title('Clusters Visualization')
plt.xlabel('MDS Dimension 1')
plt.ylabel('MDS Dimension 2')
plt.legend()
plt.show()
