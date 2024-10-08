import preprocessing
from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, squareform, pdist


def perform_kmeans_clustering(coordinates, num_clusters):
    """
    Perform K-means clustering using the 2D coordinates.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(coordinates)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    plt.figure(figsize=(12, 6))

    for i in range(num_clusters):
        cluster_points = coordinates[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')

    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')

    plt.title('K-means Clustering on 2D Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

    return labels, centroids


def plot_initial_assignments(selected_points, coordinates, centroids):
    """
    Plot the initial assignments used as a warm start for Gurobi.
    """
    plt.figure(figsize=(12, 6))
    num_clusters = len(selected_points)

    for l in range(num_clusters):
        cluster_points = coordinates[selected_points[l]]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {l}')

    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='X', label='Centroids')

    plt.title('Initial Assignments Used for Warm Start')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_kmeans_objective(distance_matrix, kmeans_labels, num_clusters):
    """
    Calculate the objective value for K-means clustering using the distance matrix.
    """
    d_max_kmeans = np.zeros(num_clusters)
    d_avg_kmeans = np.zeros(num_clusters)

    for l in range(num_clusters):
        cluster_points = np.where(kmeans_labels == l)[0]
        if len(cluster_points) > 1:
            max_dist = max(distance_matrix[i, j] for i in cluster_points for j in cluster_points if i != j)
            avg_dist = np.mean([distance_matrix[i, j] for i in cluster_points for j in cluster_points if i != j])
        else:
            max_dist = 0
            avg_dist = 0

        d_max_kmeans[l] = max_dist
        d_avg_kmeans[l] = avg_dist

    return sum(d_max_kmeans - d_avg_kmeans)


def perform_gurobi_clustering(coordinates, kmeans_labels, centroids, num_clusters, num_points_per_cluster,
                              d_max_lower_bound, d_avg_lower_bound):
    """
    Perform Gurobi-based clustering with a partial warm start using selected points closest to centroids.
    The linearization of the constraint involving d_avg is also implemented.
    """
    model = Model("MinimizeDifference")
    num_points = coordinates.shape[0]

    num_points = int(num_points)
    num_clusters = int(num_clusters)

    x = model.addVars(num_points, num_clusters, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_points, num_points, num_clusters, vtype=GRB.BINARY, name="y")
    d_max = model.addVars(num_clusters, vtype=GRB.CONTINUOUS, name="d_max")
    d_avg = model.addVars(num_clusters, vtype=GRB.CONTINUOUS, name="d_avg")
    delta = model.addVars(num_clusters, vtype=GRB.CONTINUOUS, name="delta")
    w = model.addVars(num_points, num_points, num_clusters, vtype=GRB.CONTINUOUS,
                      name="w")  # New variable for linearization

    model.setObjective(quicksum(delta[l] for l in range(num_clusters)), GRB.MINIMIZE)

    # Constraint 1: Each point must be assigned to exactly one cluster
    for i in range(num_points):
        model.addConstr(quicksum(x[i, l] for l in range(num_clusters)) == 1, name=f"Assign_{i}")

    # Warm start: selected points closest to centroids
    selected_points = select_points_closest_to_centroids(coordinates, kmeans_labels, centroids, num_points_per_cluster)

    for l, points in selected_points.items():
        for i in points:
            x[i, l].start = 1

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
                    model.addConstr(np.linalg.norm(coordinates[i] - coordinates[j]) * y[i, j, l] <= d_max[l],
                                    name=f"MaxDist_{i}_{j}_{l}")

    # Linearization of the original constraint 4
    M = np.max(
        np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2))  # Maximum distance in the distance matrix

    for l in range(num_clusters):
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    model.addConstr(w[i, j, l] <= M * y[i, j, l], name=f"Lin1_{i}_{j}_{l}")
                    model.addConstr(w[i, j, l] <= d_avg[l], name=f"Lin2_{i}_{j}_{l}")
                    model.addConstr(w[i, j, l] >= d_avg[l] - (1 - y[i, j, l]) * M, name=f"Lin3_{i}_{j}_{l}")

    # Replacing original constraint 4 with the linearized version
    for l in range(num_clusters):
        model.addConstr(
            quicksum(w[i, j, l] for i in range(num_points) for j in range(num_points) if i != j) == quicksum(
                np.linalg.norm(coordinates[i] - coordinates[j]) * y[i, j, l] for i in range(num_points) for j in
                range(num_points) if i != j), name=f"NewAvgDist_{l}")

    # Constraint 5: delta as the difference between d_max and d_avg
    for l in range(num_clusters):
        model.addConstr(d_max[l] - d_avg[l] == delta[l], name=f"Delta_{l}")

    # Adding lower bound constraints for d_max and d_avg
    for l in range(num_clusters):
        model.addConstr(d_max[l] >= d_max_lower_bound, name=f"d_max_lower_bound_{l}")
        model.addConstr(d_avg[l] >= d_avg_lower_bound, name=f"d_avg_lower_bound_{l}")

    kmeans_obj = calculate_kmeans_objective(np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2),
                                            kmeans_labels, num_clusters)
    print(f"K-means objective value: {kmeans_obj}")

    model.addConstr(quicksum(delta[l] for l in range(num_clusters)) <= kmeans_obj, name="KmeansBound")

    model.setParam('TimeLimit', 600)  # Set a 10-minute time limit
    model.optimize()

    custom_clusters = {l: [] for l in range(num_clusters)}

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        for i in range(num_points):
            for l in range(num_clusters):
                if x[i, l].X > 0.5:
                    custom_clusters[l].append(i)
        print("Clustering after Gurobi optimization:")
        for l, points in custom_clusters.items():
            print(f"Cluster {l}: Points {points}")
    else:
        print("No optimal solution found or optimization terminated early.")

    if model.status == GRB.TIME_LIMIT:
        print(f"Time limit reached. Best solution found with objective: {model.ObjVal}")

    return custom_clusters



def plot_gurobi_results(custom_clusters, coordinates):
    """
    Plot the results of Gurobi optimization clustering.
    """
    new_centroids = []
    for l in custom_clusters:
        cluster_points = coordinates[custom_clusters[l]]
        new_centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(new_centroid)

    new_centroids = np.array(new_centroids)
    plt.figure(figsize=(12, 6))
    for l in custom_clusters:
        cluster_points = coordinates[custom_clusters[l]]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {l}')

    plt.scatter(new_centroids[:, 0], new_centroids[:, 1], s=300, c='red', marker='X', label='New Centroids')

    plt.title('Gurobi Optimization Clustering on 2D Coordinates with New Centroids')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()


def find_best_k(coordinates, max_k):
    """
    Use the Elbow Method to find the best number of clusters for K-means.
    """
    sse = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(coordinates)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of squared distances (SSE)')
    plt.grid(True)
    plt.show()

    elbow_point = np.argmin(np.diff(sse, 2)) + 2  # +2 because diff reduces the length by 2

    print(f"Optimal number of clusters (elbow point): {elbow_point}")
    return elbow_point


def select_points_closest_to_centroids(coordinates, labels, centroids, num_points_per_cluster):
    """
    Selects a subset of points that are closest to their respective cluster centroids.
    """
    selected_points = {l: [] for l in range(centroids.shape[0])}

    for l in range(centroids.shape[0]):
        cluster_points = np.where(labels == l)[0]
        distances = np.linalg.norm(coordinates[cluster_points] - centroids[l], axis=1)
        closest_points = cluster_points[np.argsort(distances)[:num_points_per_cluster]]

        selected_points[l].extend(closest_points)

    return selected_points


def main():
    df = pd.read_excel('20 Clusters.xlsx', sheet_name='Clusters')
    df[['X', 'Y']] = df['Coordinates'].str.strip('()').str.split(',', expand=True).astype(float)
    coordinates = df[['X', 'Y']].values
    distances = squareform(pdist(coordinates, 'euclidean'))

    d_max_lower_bound = np.min(distances[np.nonzero(distances)])
    d_avg_lower_bound = np.mean(distances[np.nonzero(distances)])

    max_k = 20
    optimal_k = find_best_k(coordinates, max_k)

    num_points_per_cluster = 10
    num_clusters = optimal_k
    kmeans_labels, kmeans_centroids = perform_kmeans_clustering(coordinates, num_clusters)

    selected_points = select_points_closest_to_centroids(coordinates, kmeans_labels, kmeans_centroids,
                                                         num_points_per_cluster)

    plot_initial_assignments(selected_points, coordinates, kmeans_centroids)

    custom_clusters = perform_gurobi_clustering(coordinates, kmeans_labels, kmeans_centroids,
                                                num_clusters, num_points_per_cluster, d_max_lower_bound, d_avg_lower_bound)

    plot_gurobi_results(custom_clusters, coordinates)


if __name__ == "__main__":
    main()
