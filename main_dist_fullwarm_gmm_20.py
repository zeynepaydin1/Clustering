import preprocessing
from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def perform_gmm_clustering(coordinates, num_clusters):
    """
    Performs Gaussian Mixture Model (GMM) clustering using the coordinates.
    """
    gmm = GaussianMixture(n_components=num_clusters, random_state=0)
    gmm.fit(coordinates)
    labels = gmm.predict(coordinates)
    return labels

def plot_clusters_with_coordinates(clusters, coordinates):
    """
    Plots the clusters using the 2D coordinates after clustering based on the distance matrix.
    """
    plt.figure(figsize=(12, 6))
    num_clusters = len(clusters)

    for l in range(num_clusters):
        cluster_points = coordinates[clusters[l]]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {l}')

    plt.title('GMM Clustering on Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

def perform_gurobi_clustering(djlorj_matrix, initial_labels, num_clusters, d_max_lower_bound, d_avg_lower_bound, coordinates):
    """
    Performs Gurobi-based clustering using the initial clustering results (from GMM) as a warm start.
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
    #z = model.addVar(vtype=GRB.CONTINUOUS, name="z")

    # Objective 1: Minimize the sum of delta[l]
    model.setObjectiveN(quicksum(delta[l] for l in range(num_clusters)), index=0, priority=0)

    # Objective 2: Minimize z
    #model.setObjectiveN(z, index=1, priority=1)

    # Constraint 1: Each point must be assigned to exactly one cluster
    for i in range(num_points):
        model.addConstr(quicksum(x[i, l] for l in range(num_clusters)) == 1, name=f"Assign_{i}")

    # Warm start: Uses initial clustering results (GMM)
    for i in range(num_points):
        for l in range(num_clusters):
            if initial_labels[i] == l:
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

    # Modified Constraint 4: d_avg as the average distance within each cluster
    for l in range(num_clusters):
        cluster_points = [i for i in range(num_points) if initial_labels[i] == l]
        num_pairs = len(cluster_points) * (len(cluster_points) - 1)
        if num_pairs > 0:
            total_distance = quicksum(djlorj_matrix[i, j] for i in cluster_points for j in cluster_points if i != j)
            avg_distance = total_distance / num_pairs
            model.addConstr(d_avg[l] == avg_distance, name=f"AvgDist_{l}")
        else:
            # Handle single-point clusters
            model.addConstr(d_avg[l] == 0, name=f"AvgDist_{l}")

    # Constraint 5: delta as the difference between d_max and d_avg
    for l in range(num_clusters):
        model.addConstr(d_max[l] - d_avg[l] == delta[l], name=f"Delta_{l}")

    """
    # New Constraint: delta[l] <= z
    for l in range(num_clusters):
        model.addConstr(delta[l] <= z, name=f"Delta_Leq_Z_{l}")

    for l in range(num_clusters):
        model.addConstr(d_max[l] >= d_max_lower_bound, name=f"d_max_lower_bound_{l}")
        model.addConstr(d_avg[l] >= d_avg_lower_bound, name=f"d_avg_lower_bound_{l}")
    """
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
            if len(custom_clusters[l]) > 0:
                print(f"Cluster {l}: d_avg = {d_avg[l].X}")
                cluster_points = custom_clusters[l]
                centroid_x = np.mean([coordinates[i, 0] for i in cluster_points])
                centroid_y = np.mean([coordinates[i, 1] for i in cluster_points])
                print(f"Cluster {l} Centroid: ({centroid_x}, {centroid_y})")
    else:
        print("No optimal solution found or optimization terminated early.")

    changes = np.where(initial_labels != new_labels)[0]
    if len(changes) > 0:
        print(f"Points with changed assignments: {changes}")
    else:
        print("No points' assignments were changed by Gurobi optimization.")

    return custom_clusters

def find_best_gmm_k(coordinates, max_k=10):
    """
    Uses the Silhouette Method to find the best number of clusters for GMM.
    """
    silhouette_scores = []

    for k in range(2, max_k + 1):
        labels = perform_gmm_clustering(coordinates, k)
        score = silhouette_score(coordinates, labels)
        silhouette_scores.append(score)
        print(f"Silhouette score for {k} clusters: {score}")

    optimal_k = np.argmax(silhouette_scores) + 2
    print(f"Optimal number of clusters: {optimal_k}")
    return optimal_k

def calculate_avg_travel_time(djlorj_matrix, clusters):
    """
    Calculate the average travel time between any two clusters based on the djlorj matrix.
    """
    clusters = list(clusters)
    num_clusters = len(clusters)
    total_time = 0
    count = 0

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            cluster_i = clusters[i]
            cluster_j = clusters[j]
            total_distance = 0
            num_pairs = len(cluster_i) * len(cluster_j)
            if num_pairs > 0:
                total_distance = sum(djlorj_matrix[p1, p2] for p1 in cluster_i for p2 in cluster_j)
                avg_distance = total_distance / num_pairs
                total_time += avg_distance
                count += 1

    avg_travel_time = total_time / count if count > 0 else 0
    return avg_travel_time

def main():
    df = pd.read_excel('coord_8_20.xlsx', sheet_name='coordination of centers (20)')
    coordinates = df.iloc[0:, 1:3].values
    djlorj_matrix = preprocessing.preprocess("20pts.txt").values
    d_max_lower_bound = np.min(djlorj_matrix[np.nonzero(djlorj_matrix)])
    d_avg_lower_bound = np.mean(djlorj_matrix[np.nonzero(djlorj_matrix)])

    optimal_k = find_best_gmm_k(coordinates, max_k=10)

    num_clusters = optimal_k
    gmm_labels = perform_gmm_clustering(coordinates, num_clusters)
    plot_clusters_with_coordinates({l: np.where(gmm_labels == l)[0] for l in range(num_clusters)}, coordinates)

    custom_clusters = perform_gurobi_clustering(djlorj_matrix, gmm_labels, num_clusters,
                                                d_max_lower_bound, d_avg_lower_bound, coordinates)

    plot_clusters_with_coordinates(custom_clusters, coordinates)

    avg_travel_time = calculate_avg_travel_time(djlorj_matrix, custom_clusters.values())
    print(f"Average Travel Time between any two clusters: {avg_travel_time:.2f} minutes")


if __name__ == "__main__":
    main()
