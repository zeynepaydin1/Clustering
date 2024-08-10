import preprocessing
from gurobipy import Model, GRB, quicksum
import numpy as np

distance_matrix_cleaned = preprocessing.preprocess("djlorj.txt")
distance_matrix = np.array(distance_matrix_cleaned)
distance_matrix = (distance_matrix + distance_matrix.T) / 2 #symmetrized distance_matrix
num_points = len(distance_matrix)
num_clusters = 4  #from elbow plot

#Gurobi model
model = Model("MinimizeDifference")

#Variables
x = model.addVars(num_points, num_clusters, vtype=GRB.BINARY, name="x")
y = model.addVars(num_points, num_points, num_clusters, vtype=GRB.BINARY, name="y")
d_max = model.addVars(num_clusters, vtype=GRB.CONTINUOUS, name="d_max")
d_avg = model.addVars(num_clusters, vtype=GRB.CONTINUOUS, name="d_avg")
delta = model.addVars(num_clusters, vtype=GRB.CONTINUOUS, name="delta")

# Objective: Minimize the sum of differences between max and average distances in clusters
model.setObjective(quicksum(delta[l] for l in range(num_clusters)), GRB.MINIMIZE)

# Constraint 1: Each point must be assigned to exactly one cluster
for i in range(num_points):
    model.addConstr(quicksum(x[i, l] for l in range(num_clusters)) == 1, name=f"Assign_{i}")

# Constraint 2: y_ijl based on x_il and x_jl
for i in range(num_points):
    for j in range(num_points):
        if i != j:  # Avoid self-pairing
            for l in range(num_clusters):
                model.addConstr(y[i, j, l] <= x[i, l], name=f"Link1_{i}_{j}_{l}")
                model.addConstr(y[i, j, l] <= x[j, l], name=f"Link2_{i}_{j}_{l}")
                model.addConstr(x[i, l] + x[j, l] - 1 <= y[i, j, l], name=f"Link3_{i}_{j}_{l}")

# Constraint 3: d_max is at least the distance between any two points in the same cluster
for l in range(num_clusters):
    for i in range(num_points):
        for j in range(num_points):
            if i != j:  # Avoid self-pairing
                model.addConstr(distance_matrix[i][j] * y[i, j, l] <= d_max[l], name=f"MaxDist_{i}_{j}_{l}")

# Constraint 4: d_avg as the average distance within each cluster
for l in range(num_clusters):
    total_distance = quicksum(distance_matrix[i][j] * y[i, j, l] for i in range(num_points) for j in range(num_points) if i != j)
    total_pairs = quicksum(y[i, j, l] for i in range(num_points) for j in range(num_points) if i != j)
    model.addConstr(total_distance == d_avg[l] * total_pairs, name=f"AvgDist_{l}")

# Constraint 5: delta as the difference between d_max and d_avg
for l in range(num_clusters):
    model.addConstr(d_max[l] - d_avg[l] == delta[l], name=f"Delta_{l}")

model.optimize()

if model.status == GRB.OPTIMAL:
    clusters = {l: [] for l in range(num_clusters)}
    for i in range(num_points):
        for l in range(num_clusters):
            if x[i, l].x > 0.5:
                clusters[l].append(i)

    print("Optimal Clustering:")
    for l, points in clusters.items():
        print(f"Cluster {l}: Points {points}")

else:
    print("No optimal solution found.")
