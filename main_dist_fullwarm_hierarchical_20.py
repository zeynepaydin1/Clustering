import preprocessing
from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

def perform_hierarchical_clustering(djlorj_matrix, num_clusters):
    """
    Performs hierarchical clustering using the distance matrix.
    """
    Z = linkage(djlorj_matrix, method='complete')
    labels = fcluster(Z, num_clusters, criterion='maxclust')
    return labels - 1  # Adjusting labels to start from 0

def plot_clusters_with_coordinates(clusters, coordinates):
    """
    Plots the clusters using the 2D coordinates after clustering based on the distance matrix.
    """
    plt.figure(figsize=(12, 6))
    num_clusters = len(clusters)

    for l in range(num_clusters):
        cluster_points = coordinates[clusters[l]]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {l}')

    plt.title('Hierarchical Clustering on Distance Matrix with Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

def perform_gurobi_clustering(djlorj_matrix, hierarchical_labels, num_clusters, d_max_lower_bound, d_avg_lower_bound,
                              coordinates):
    """
    Performs Gurobi-based clustering using the full hierarchical clustering results as a warm start.
    Also, prints d_avg and the locations of the cluster centroids.
    """
    model = Model("MinimizeDifference")
    num_points = int(djlorj_matrix.shape[0])
    num_clusters = int(num_clusters)

    x = model.addVars(num_points, num_clusters, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_points, num_points, num_clusters, vtype=GRB.BINARY, name="y")
    d_max = model.addVars(num_clusters, vtype=GRB.CONTINUOUS, name="d_max")
    d_avg = model.addVars(num_clusters, vtype=GRB.CONTINUOUS, name="d_avg")
    delta = model.addVars(num_clusters, vtype=GRB.CONTINUOUS, name="delta")
    z = model.addVar(vtype=GRB.CONTINUOUS, name="z")

    # Objective 1: Minimize the sum of delta[l]
    model.setObjectiveN(quicksum(delta[l] for l in range(num_clusters)), index=0, priority=1)

    # Objective 2: Minimize z
    model.setObjectiveN(z, index=1, priority=0)

    # Constraint 1: Each point must be assigned to exactly one cluster
    for i in range(num_points):
        model.addConstr(quicksum(x[i, l] for l in range(num_clusters)) == 1, name=f"Assign_{i}")

    # Warm start: Uses hierarchical clustering results
    for i in range(num_points):
        for l in range(num_clusters):
            if hierarchical_labels[i] == l:
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

    # New Constraint: delta[l] <= z
    for l in range(num_clusters):
        model.addConstr(delta[l] <= z, name=f"Delta_Leq_Z_{l}")

    for l in range(num_clusters):
        model.addConstr(d_max[l] >= d_max_lower_bound, name=f"d_max_lower_bound_{l}")
        model.addConstr(d_avg[l] >= d_avg_lower_bound, name=f"d_avg_lower_bound_{l}")

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

        for l in range(num_clusters):
            print(f"Cluster {l}: d_avg = {d_avg[l].X}")
            cluster_points = custom_clusters[l]
            centroid_x = np.mean([coordinates[i, 0] for i in cluster_points])
            centroid_y = np.mean([coordinates[i, 1] for i in cluster_points])
            print(f"Cluster {l} Centroid: ({centroid_x}, {centroid_y})")
    else:
        print("No optimal solution found or optimization terminated early.")

    changes = np.where(hierarchical_labels != new_labels)[0]
    if len(changes) > 0:
        print(f"Points with changed assignments: {changes}")
    else:
        print("No points' assignments were changed by Gurobi optimization.")

    return custom_clusters

def find_best_k(djlorj_matrix, max_k=10):
    """
    Uses the Silhouette Method to find the best number of clusters for hierarchical clustering.
    """
    Z = linkage(djlorj_matrix, method='complete')
    silhouette_scores = []

    for k in range(2, max_k + 1):
        labels = fcluster(Z, k, criterion='maxclust')
        score = silhouette_score(djlorj_matrix, labels, metric='precomputed')
        silhouette_scores.append(score)
        print(f"Silhouette score for {k} clusters: {score}")

    optimal_k = np.argmax(silhouette_scores) + 2  # +2 because range starts from 2
    print(f"Optimal number of clusters: {optimal_k}")
    return optimal_k

def main():
    df = pd.read_excel('coord_8_20.xlsx', sheet_name='coordination of centers (20)')
    coordinates = df.iloc[0:, 1:3].values
    djlorj_matrix = preprocessing.preprocess("20pts.txt").values
    d_max_lower_bound = np.min(djlorj_matrix[np.nonzero(djlorj_matrix)])
    d_avg_lower_bound = np.mean(djlorj_matrix[np.nonzero(djlorj_matrix)])

    optimal_k = find_best_k(djlorj_matrix, max_k=10)

    num_clusters = optimal_k
    hierarchical_labels = perform_hierarchical_clustering(djlorj_matrix, num_clusters)

    plot_clusters_with_coordinates({l: np.where(hierarchical_labels == l)[0] for l in range(num_clusters)}, coordinates)
    custom_clusters = perform_gurobi_clustering(djlorj_matrix, hierarchical_labels, num_clusters,
                                                d_max_lower_bound, d_avg_lower_bound, coordinates)

    plot_clusters_with_coordinates(custom_clusters, coordinates)


if __name__ == "__main__":
    main()
