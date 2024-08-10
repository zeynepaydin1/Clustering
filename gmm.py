import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import preprocessing

time_matrix = preprocessing.preprocess("djlorj.txt")

pca = PCA(n_components=2)
points = pca.fit_transform(time_matrix)

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(points)
labels = gmm.predict(points)

plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')
plt.title('GMM Clustering')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.show()
