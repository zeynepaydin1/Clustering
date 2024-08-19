import preprocessing
from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist


def perform_kmeans_clustering(djlorj_matrix, num_clusters):
    """
    Performs K-means clustering using the distance matrix.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(djlorj_matrix)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    return labels, centroids


def plot_clusters_with_coordinates(clusters, coordinates):
    """
    Plots the clusters using the 2D coordinates after clustering based on the distance matrix.
    """
    plt.figure(figsize=(12, 6))
    num_clusters = len(clusters)

    new_centroids = []
    for l in range(num_clusters):
        cluster_points = coordinates[clusters[l]]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {l}')
        new_centroids.append(np.mean(cluster_points, axis=0))

    new_centroids = np.array(new_centroids)
    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], s=300, c='red', marker='X', label='Centroids')

    plt.title('Gurobi Optimization Clustering on Distance Matrix with New Centroids')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_kmeans_clusters_with_coordinates(clusters, coordinates):
    """
    Plots the clusters using the 2D coordinates after clustering based on the distance matrix.
    """
    plt.figure(figsize=(12, 6))
    num_clusters = len(clusters)

    new_centroids = []
    for l in range(num_clusters):
        cluster_points = coordinates[clusters[l]]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {l}')
        new_centroids.append(np.mean(cluster_points, axis=0))

    new_centroids = np.array(new_centroids)
    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], s=300, c='black', marker='X', label='Centroids')

    plt.title('K-means Clustering on 2D Coordinates based on Distance Matrix')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_kmeans_objective(djlorj_matrix, kmeans_labels, num_clusters):
    """
    Calculates the objective value for K-means clustering using the djlorj matrix.
    """
    d_max_kmeans = np.zeros(num_clusters)
    d_avg_kmeans = np.zeros(num_clusters)

    for l in range(num_clusters):
        cluster_points = np.where(kmeans_labels == l)[0]
        if len(cluster_points) > 1:
            max_dist = max(djlorj_matrix[i, j] for i in cluster_points for j in cluster_points if i != j)
            avg_dist = np.mean([djlorj_matrix[i, j] for i in cluster_points for j in cluster_points if i != j])
        else:
            max_dist = 0
            avg_dist = 0

        d_max_kmeans[l] = max_dist
        d_avg_kmeans[l] = avg_dist

    return sum(d_max_kmeans - d_avg_kmeans)


def perform_gurobi_clustering(djlorj_matrix, kmeans_labels, num_clusters, d_max_lower_bound, d_avg_lower_bound):
    """
    Performs Gurobi-based clustering using the full K-means clustering results as a warm start.
    """
    model = Model("MinimizeDifference")
    num_points = int(djlorj_matrix.shape[0])
    num_clusters = int(num_clusters)

    x = model.addVars(num_points, num_clusters, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_points, num_points, num_clusters, vtype=GRB.BINARY, name="y")
    d_max = model.addVars(num_clusters, vtype=GRB.CONTINUOUS, name="d_max")
    d_avg = model.addVars(num_clusters, vtype=GRB.CONTINUOUS, name="d_avg")
    delta = model.addVars(num_clusters, vtype=GRB.CONTINUOUS, name="delta")

    model.setObjective(quicksum(delta[l] for l in range(num_clusters)), GRB.MINIMIZE)

    # Constraint 1: Each point must be assigned to exactly one cluster
    for i in range(num_points):
        model.addConstr(quicksum(x[i, l] for l in range(num_clusters)) == 1, name=f"Assign_{i}")

    # Warm start: Uses K-means clustering results
    for i in range(num_points):
        for l in range(num_clusters):
            if kmeans_labels[i] == l:
                x[i, l].start = 1
            else:
                x[i, l].start = 0

    # Constraint 2: y_ijl based on x_il and x_jl
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                for l in range(num_clusters):
                    model.addConstr(y[i, j, l] <= x[i, l], name=f"Link1_{i}_{j}_{l}")
                    model.addConstr(y[i, j, l] <= x[j, l], name=f"Link2_{i}_{j}_{l}")
                    model.addConstr(x[i, l] + x[j, l] - 1 <= y[i, j, l], name=f"Link3_{i}_{j}_{l}")

    # Constraint 3: d_max is at least the distance between any two points in the same cluster
    for l in range(num_clusters):
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    model.addConstr(djlorj_matrix[i, j] * y[i, j, l] <= d_max[l], name=f"MaxDist_{i}_{j}_{l}")

    # Linearized Constraint 4: d_avg as the average distance within each cluster
    M = np.max(djlorj_matrix)
    w = model.addVars(num_points, num_points, num_clusters, vtype=GRB.CONTINUOUS, name="w")

    for l in range(num_clusters):
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    model.addConstr(w[i, j, l] <= M * y[i, j, l], name=f"Linearize1_{i}_{j}_{l}")
                    model.addConstr(w[i, j, l] <= d_avg[l], name=f"Linearize2_{i}_{j}_{l}")
                    model.addConstr(w[i, j, l] >= d_avg[l] - (1 - y[i, j, l]) * M, name=f"Linearize3_{i}_{j}_{l}")

    for l in range(num_clusters):
        total_distance = quicksum(w[i, j, l] for i in range(num_points) for j in range(num_points) if i != j)
        total_pairs = quicksum(y[i, j, l] for i in range(num_points) for j in range(num_points) if i != j)
        model.addConstr(total_distance == d_avg[l] * total_pairs, name=f"AvgDist_{l}")

    # Constraint 5: delta as the difference between d_max and d_avg
    for l in range(num_clusters):
        model.addConstr(d_max[l] - d_avg[l] == delta[l], name=f"Delta_{l}")

    for l in range(num_clusters):
        model.addConstr(d_max[l] >= d_max_lower_bound, name=f"d_max_lower_bound_{l}")
        model.addConstr(d_avg[l] >= d_avg_lower_bound, name=f"d_avg_lower_bound_{l}")

    kmeans_obj = calculate_kmeans_objective(djlorj_matrix, kmeans_labels, num_clusters)
    print(f"K-means objective value: {kmeans_obj}")

    model.addConstr(quicksum(delta[l] for l in range(num_clusters)) <= kmeans_obj, name="KmeansBound")

    model.optimize()

    custom_clusters = {l: [] for l in range(num_clusters)}
    new_labels = np.zeros(num_points, dtype=int)
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        for i in range(num_points):
            for l in range(num_clusters):
                if x[i, l].X > 0.5:
                    custom_clusters[l].append(i)
                    new_labels[i] = l
        print("Clustering after Gurobi optimization:")
        for l, points in custom_clusters.items():
            print(f"Cluster {l}: Points {points}")
    else:
        print("No optimal solution found or optimization terminated early.")

    changes = np.where(kmeans_labels != new_labels)[0]
    if len(changes) > 0:
        print(f"Points with changed assignments: {changes}")
    else:
        print("No points' assignments were changed by Gurobi optimization.")

    return custom_clusters


def find_best_k(djlorj_matrix, max_k):
    """
    Uses the Elbow Method to find the best number of clusters for K-means.
    """
    sse = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(djlorj_matrix)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared distances (SSE)')
    plt.grid(True)
    plt.show()

    elbow_point = np.argmin(np.diff(sse, 2)) + 2  #+2 because diff reduces the length by 2

    print(f"Optimal number of clusters (elbow point): {elbow_point}")
    return elbow_point


def main():
    coordinates = pd.read_excel('20 Clusters.xlsx', sheet_name='Clusters')
    coordinates[['X', 'Y']] = coordinates['Coordinates'].str.strip('()').str.split(',', expand=True).astype(float)
    coordinates = coordinates[['X', 'Y']].values
    djlorj_matrix = preprocessing.preprocess("djlorj.txt").values
    d_max_lower_bound = np.min(djlorj_matrix[np.nonzero(djlorj_matrix)])
    d_avg_lower_bound = np.mean(djlorj_matrix[np.nonzero(djlorj_matrix)])

    max_k = 20
    optimal_k = find_best_k(djlorj_matrix, max_k)

    num_clusters = optimal_k
    kmeans_labels, kmeans_centroids = perform_kmeans_clustering(djlorj_matrix, num_clusters)

    plot_kmeans_clusters_with_coordinates({l: np.where(kmeans_labels == l)[0] for l in range(num_clusters)}, coordinates)
    custom_clusters = perform_gurobi_clustering(djlorj_matrix, kmeans_labels, num_clusters,
                                                d_max_lower_bound, d_avg_lower_bound)

    plot_clusters_with_coordinates(custom_clusters, coordinates)


if __name__ == "__main__":
    main()


