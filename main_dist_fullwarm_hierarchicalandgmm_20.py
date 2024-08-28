import preprocessing
from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances

def perform_hierarchical_clustering(djlorj_matrix, num_clusters):
    """
    Performs hierarchical clustering using the distance matrix.
    """
    Z = linkage(djlorj_matrix, method='complete')
    labels = fcluster(Z, num_clusters, criterion='maxclust')
    return labels  # No need to subtract 1 since we're using 1-based indexing

def calculate_hierarchical_clustering_objective(djlorj_matrix, hierarchical_labels, num_clusters):
    """
    Calculate the objective value from the hierarchical clustering results.
    The objective is the sum of differences between the maximum and average distances within each cluster.
    """
    objective_value = 0

    for l in range(1, num_clusters + 1):  # Adjust range for 1-based indexing
        cluster_points = [i for i in range(len(hierarchical_labels)) if hierarchical_labels[i] == l]
        if len(cluster_points) < 2:
            continue

        max_distance = 0
        total_distance = 0
        count = 0

        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                p1, p2 = cluster_points[i], cluster_points[j]
                distance = djlorj_matrix[p1, p2]
                max_distance = max(max_distance, distance)
                total_distance += distance
                count += 1

        avg_distance = total_distance / count if count > 0 else 0
        objective_value += max_distance - avg_distance

    return objective_value

def perform_gurobi_clustering(djlorj_matrix, hierarchical_labels, num_clusters, d_max_lower_bound, d_avg_lower_bound,
                              coordinates, hierarchical_objective_value):
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
        model.addConstr(quicksum(x[i, l] for l in range(num_clusters)) == 1, name=f"Assign_{i+1}")

    # Warm start: Uses hierarchical clustering results
    for i in range(num_points):
        for l in range(num_clusters):
            if hierarchical_labels[i] == l + 1:  # Adjust for 1-based indexing
                x[i, l].start = 1
            else:
                x[i, l].start = 0

    # Constraint 2: y_ijl based on x_il and x_jl
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                for l in range(num_clusters):
                    model.addConstr(y[i, j, l] <= x[i, l], name=f"Link1_{i+1}_{j+1}_{l+1}")
                    model.addConstr(y[i, j, l] <= x[j, l], name=f"Link2_{i+1}_{j+1}_{l+1}")
                    model.addConstr(x[i, l] + x[j, l] - 1 <= y[i, j, l], name=f"Link3_{i+1}_{j+1}_{l+1}")

    # Constraint 3: d_max is at least the distance between any two points in the same cluster
    for l in range(num_clusters):
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    model.addConstr(djlorj_matrix[i, j] * y[i, j, l] <= d_max[l], name=f"MaxDist_{i+1}_{j+1}_{l+1}")

    # Linearized Constraint 4: d_avg as the average distance within each cluster
    M = np.max(djlorj_matrix)
    w = model.addVars(num_points, num_points, num_clusters, vtype=GRB.CONTINUOUS, name="w")

    for l in range(num_clusters):
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    model.addConstr(w[i, j, l] <= M * y[i, j, l], name=f"Linearize1_{i+1}_{j+1}_{l+1}")
                    model.addConstr(w[i, j, l] <= d_avg[l], name=f"Linearize2_{i+1}_{j+1}_{l+1}")
                    model.addConstr(w[i, j, l] >= d_avg[l] - (1 - y[i, j, l]) * M, name=f"Linearize3_{i+1}_{j+1}_{l+1}")

    for l in range(num_clusters):
        total_distance = quicksum(w[i, j, l] for i in range(num_points) for j in range(num_points) if i != j)
        total_pairs = quicksum(y[i, j, l] for i in range(num_points) for j in range(num_points) if i != j)
        model.addConstr(total_distance == d_avg[l] * total_pairs, name=f"AvgDist_{l+1}")

    # Constraint 5: delta as the difference between d_max and d_avg
    for l in range(num_clusters):
        model.addConstr(d_max[l] - d_avg[l] == delta[l], name=f"Delta_{l+1}")

    # New Constraint: delta[l] <= z
    for l in range(num_clusters):
        model.addConstr(delta[l] <= z, name=f"Delta_Leq_Z_{l+1}")

    # Hierarchical Clustering Objective Constraint
    model.addConstr(quicksum(delta[l] for l in range(num_clusters)) <= hierarchical_objective_value,
                    name="HierarchicalObjectiveConstraint")

    for l in range(num_clusters):
        model.addConstr(d_max[l] >= d_max_lower_bound, name=f"d_max_lower_bound_{l+1}")
        model.addConstr(d_avg[l] >= d_avg_lower_bound, name=f"d_avg_lower_bound_{l+1}")

    model.optimize()

    custom_clusters = {l + 1: [] for l in range(num_clusters)}  # Adjust for 1-based indexing
    new_labels = np.zeros(num_points, dtype=int)
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        for i in range(num_points):
            for l in range(num_clusters):
                if x[i, l].X > 0.5:
                    custom_clusters[l + 1].append(i + 1)  # Adjust for 1-based indexing
                    new_labels[i] = l + 1  # Adjust for 1-based indexing
        print("Clustering after Gurobi optimization:")
        for l, points in custom_clusters.items():
            print(f"Cluster {l}: Points {points}")

        for l in range(num_clusters):
            print(f"Cluster {l+1}: d_avg = {d_avg[l].X}")
            cluster_points = custom_clusters[l+1]
            centroid_x = np.mean([coordinates[i-1, 0] for i in cluster_points])  # Adjust for 1-based indexing
            centroid_y = np.mean([coordinates[i-1, 1] for i in cluster_points])  # Adjust for 1-based indexing
            print(f"Cluster {l+1} Centroid: ({centroid_x}, {centroid_y})")
    else:
        print("No optimal solution found or optimization terminated early.")

    changes = np.where(hierarchical_labels != new_labels)[0]
    if len(changes) > 0:
        print(f"Points with changed assignments: {changes+1}")  # Adjust for 1-based indexing
    else:
        print("No points' assignments were changed by Gurobi optimization.")

    return custom_clusters

def calculate_in_cluster_avg_travel_time(djlorj_matrix, custom_clusters):
    """
    Calculate the InClusterAvgTravelTime for each cluster based on the djlorj_matrix.
    """
    in_cluster_avg_travel_time = {}

    for cluster_id, cluster_points in custom_clusters.items():
        if len(cluster_points) < 2:
            in_cluster_avg_travel_time[cluster_id] = 0
            continue

        total_distance = 0
        count = 0
        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                p1, p2 = cluster_points[i] - 1, cluster_points[j] - 1  # Adjust for 1-based indexing
                total_distance += djlorj_matrix[p1, p2] + djlorj_matrix[p2, p1]
                count += 2  # We consider both (p1, p2) and (p2, p1)

        in_cluster_avg_travel_time[cluster_id] = total_distance / count if count > 0 else 0

    return in_cluster_avg_travel_time


def find_best_k_hierarchical(djlorj_matrix, max_k=10):
    """
    Uses the Silhouette Method to find the best number of clusters for hierarchical clustering.
    """
    Z = linkage(djlorj_matrix, method='complete')
    silhouette_scores = []

    for k in range(2, max_k + 1):
        labels = fcluster(Z, k, criterion='maxclust')
        score = silhouette_score(djlorj_matrix, labels, metric='precomputed')
        silhouette_scores.append(score)
        print(f"Silhouette score for {k} clusters (Hierarchical): {score}")

    optimal_k = np.argmax(silhouette_scores) + 2  # +2 because range starts from 2
    print(f"Optimal number of clusters (Hierarchical): {optimal_k}")
    return optimal_k


def find_best_k_gmm(djlorj_matrix, max_k=10):
    """
    Uses the Bayesian Information Criterion (BIC) or Silhouette Method to find the best number of clusters for GMM.
    """
    bic_scores = []
    silhouette_scores = []

    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(djlorj_matrix)
        labels = gmm.predict(djlorj_matrix)
        bic = gmm.bic(djlorj_matrix)
        bic_scores.append(bic)

        score = silhouette_score(djlorj_matrix, labels, metric='precomputed')
        silhouette_scores.append(score)

        print(f"BIC for {k} clusters (GMM): {bic}")
        print(f"Silhouette score for {k} clusters (GMM): {score}")

    # Select the optimal K based on the lowest BIC score or highest silhouette score
    optimal_k_bic = np.argmin(bic_scores) + 2  # +2 because range starts from 2
    optimal_k_silhouette = np.argmax(silhouette_scores) + 2  # +2 because range starts from 2

    print(f"Optimal number of clusters (GMM - BIC): {optimal_k_bic}")
    print(f"Optimal number of clusters (GMM - Silhouette): {optimal_k_silhouette}")

    # Optionally, return the one based on BIC or Silhouette
    # Here, we return the one with the highest silhouette score
    return optimal_k_silhouette

def perform_gmm(djlorj_matrix, num_clusters):
    """
    Performs Gaussian Mixture Model (GMM) clustering using the distance matrix.
    """
    # Convert the distance matrix to a 2D coordinate representation using MDS or similar technique
    # Here, we assume the djlorj_matrix is already in a suitable format for GMM
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    gmm.fit(djlorj_matrix)
    labels = gmm.predict(djlorj_matrix)
    return labels + 1  # Adjusting labels to start from 1 for consistency

def calculate_objective_gmm(djlorj_matrix, gmm_labels, num_clusters):
    """
    Calculate the objective value from the GMM clustering results.
    The objective is the sum of differences between the maximum and average distances within each cluster.
    """
    objective_value = 0

    for l in range(1, num_clusters + 1):  # Adjust range for 1-based indexing
        cluster_points = [i for i in range(len(gmm_labels)) if gmm_labels[i] == l]
        if len(cluster_points) < 2:
            continue

        max_distance = 0
        total_distance = 0
        count = 0

        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                p1, p2 = cluster_points[i], cluster_points[j]
                distance = djlorj_matrix[p1, p2]
                max_distance = max(max_distance, distance)
                total_distance += distance
                count += 1

        avg_distance = total_distance / count if count > 0 else 0
        objective_value += max_distance - avg_distance

    return objective_value


def plot_clusters_with_coordinates(clusters, coordinates):
    """
    Plots the clusters using the 2D coordinates after clustering based on the distance matrix.
    Also plots the centroids for each cluster.
    """
    plt.figure(figsize=(12, 6))
    num_clusters = len(clusters)

    for l in range(1, num_clusters + 1):  # Adjust loop to start from 1
        cluster_points = coordinates[np.array(clusters[l]) - 1]  # Adjust for 1-based indexing
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {l}')

        # Calculate and plot the centroid for each cluster
        centroid_x = np.mean(cluster_points[:, 0])
        centroid_y = np.mean(cluster_points[:, 1])
        plt.scatter(centroid_x, centroid_y, marker='x', color='red', s=100, label=f'Centroid {l}')

    plt.title('Clustering on Distance Matrix with Coordinates and Centroids')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():

    df = pd.read_excel('coord_8_20.xlsx', sheet_name='coordination of centers (20)')
    coordinates = df.iloc[1:, 1:3].values  # Exclude the first location from coordinates
    djlorj_matrix = preprocessing.preprocess("20pts.txt").values[1:, 1:]  # Exclude the first location from the distance matrix

    """
    coordinates = pd.read_excel('20 Clusters.xlsx', sheet_name='Clusters')
    coordinates[['X', 'Y']] = coordinates['Coordinates'].str.strip('()').str.split(',', expand=True).astype(float)
    coordinates = coordinates[['X', 'Y']].values

    djlorj_matrix = preprocessing.preprocess("djlorj.txt").values"""

    d_max_lower_bound = np.min(djlorj_matrix[np.nonzero(djlorj_matrix)])
    d_avg_lower_bound = np.mean(djlorj_matrix[np.nonzero(djlorj_matrix)])

    # Find best K for Hierarchical Clustering
    optimal_k_hierarchical = find_best_k_hierarchical(djlorj_matrix, max_k=10)

    # Find best K for GMM Clustering
    optimal_k_gmm = find_best_k_gmm(djlorj_matrix, max_k=10)

    # Perform Hierarchical Clustering
    hierarchical_labels = perform_hierarchical_clustering(djlorj_matrix, optimal_k_hierarchical)
    hierarchical_objective_value = calculate_hierarchical_clustering_objective(djlorj_matrix, hierarchical_labels, optimal_k_hierarchical)

    # Perform GMM Clustering
    gmm_labels = perform_gmm(djlorj_matrix, optimal_k_gmm)
    gmm_objective_value = calculate_objective_gmm(djlorj_matrix, gmm_labels, optimal_k_gmm)

    print(f"Hierarchical Clustering Objective Value: {hierarchical_objective_value}")
    print(f"GMM Clustering Objective Value: {gmm_objective_value}")

    # Use the method with the better objective value
    if gmm_objective_value < hierarchical_objective_value:
        labels_to_use = gmm_labels
        best_objective_value = gmm_objective_value
        print("Using GMM labels for Gurobi optimization.")
    else:
        labels_to_use = hierarchical_labels
        best_objective_value = hierarchical_objective_value
        print("Using Hierarchical labels for Gurobi optimization.")

    plot_clusters_with_coordinates({l: np.where(labels_to_use == l)[0] + 1 for l in range(1, max(optimal_k_hierarchical, optimal_k_gmm) + 1)}, coordinates)
    custom_clusters = perform_gurobi_clustering(djlorj_matrix, labels_to_use, max(optimal_k_hierarchical, optimal_k_gmm),
                                                d_max_lower_bound, d_avg_lower_bound, coordinates, best_objective_value)

    plot_clusters_with_coordinates(custom_clusters, coordinates)

    # Calculate InClusterAvgTravelTime for each cluster
    in_cluster_avg_travel_time = calculate_in_cluster_avg_travel_time(djlorj_matrix, custom_clusters)
    for cluster_id, avg_time in in_cluster_avg_travel_time.items():
        print(f"Cluster {cluster_id}: InClusterAvgTravelTime = {avg_time:.2f} minutes")


if __name__ == "__main__":
    main()
